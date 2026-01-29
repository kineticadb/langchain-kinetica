##
# Copyright (c) 2024, Chad Juliano, Kinetica DB Inc.
##
"""Kinetica SQL generation LLM API."""

import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from re import Pattern
from typing import Any, cast, override

from gpudb import GPUdb
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.utils import pre_init
from pydantic import BaseModel, ConfigDict, Field

LOG = logging.getLogger(__name__)

# Kinetica pydantic API datatypes


class _KdtSuggestContext(BaseModel):
    """pydantic API request type."""

    table: str | None = Field(default=None, title="Name of table")
    description: str | None = Field(default=None, title="Table description")
    columns: list[str] = Field(default=[], title="Table columns list")
    rules: list[str] | None = Field(
        default=None, title="Rules that apply to the table."
    )
    samples: dict | None = Field(
        default=None, title="Samples that apply to the entire context."
    )

    def to_system_str(self) -> str:
        lines = []
        lines.append(f"CREATE TABLE {self.table} AS")
        lines.append("(")

        if not self.columns or len(self.columns) == 0:
            msg = "columns list can't be null."
            raise ValueError(msg)

        columns = []
        for column in self.columns:
            column_new = column.replace('"', "").strip()
            columns.append(f"   {column_new}")
        lines.append(",\n".join(columns))
        lines.append(");")

        if self.description:
            lines.append(f"COMMENT ON TABLE {self.table} IS '{self.description}';")

        if self.rules and len(self.rules) > 0:
            lines.append(
                f"-- When querying table {self.table} the following rules apply:"
            )
            lines.extend(f"-- * {rule}" for rule in self.rules)

        return "\n".join(lines)


class _KdtSuggestPayload(BaseModel):
    """pydantic API request type."""

    question: str | None = None
    context: list[_KdtSuggestContext]

    def get_system_str(self) -> str:
        lines = []
        for table_context in self.context:
            if table_context.table is None:
                continue
            context_str = table_context.to_system_str()
            lines.append(context_str)
        return "\n\n".join(lines)

    def get_messages(self) -> list[dict]:
        messages = []
        for context in self.context:
            if context.samples is None:
                continue
            for question, answer in context.samples.items():
                # unescape double quotes
                answer_new = answer.replace("''", "'")

                messages.append({"role": "user", "content": question or ""})
                messages.append({"role": "assistant", "content": answer_new})
        return messages

    def to_completion(self) -> dict:
        messages = []
        messages.append({"role": "system", "content": self.get_system_str()})
        messages.extend(self.get_messages())
        messages.append({"role": "user", "content": self.question or ""})
        return {"messages": messages}


class _KdtoSuggestRequest(BaseModel):
    """pydantic API request type."""

    payload: _KdtSuggestPayload


class _KdtMessage(BaseModel):
    """pydantic API response type."""

    role: str = Field(default="", title="One of [user|assistant|system]")
    content: str


class _KdtChoice(BaseModel):
    """pydantic API response type."""

    index: int
    message: _KdtMessage | None = Field(default=None, title="The generated SQL")
    finish_reason: str


class _KdtUsage(BaseModel):
    """pydantic API response type."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _KdtSqlResponse(BaseModel):
    """pydantic API response type."""

    id: str
    object: str
    created: int
    model: str
    choices: list[_KdtChoice]
    usage: _KdtUsage
    prompt: str = Field(default="", title="The input question")


class _KdtCompletionResponse(BaseModel):
    """pydantic API response type."""

    status: str
    data: _KdtSqlResponse


class _KineticaLlmFileContextParser:
    """Parser for Kinetica LLM context datafiles."""

    # parse line into a dict containing role and content
    PARSER: Pattern = re.compile(r"^<\|(?P<role>\w+)\|>\W*(?P<content>.*)$", re.DOTALL)

    @classmethod
    def _removesuffix(cls, text: str, suffix: str) -> str:
        if suffix and text.endswith(suffix):
            return text[: -len(suffix)]
        return text

    @classmethod
    def parse_dialogue_file(cls, input_file: os.PathLike) -> dict:
        path = Path(input_file)
        # schema = path.name.removesuffix(".txt") python 3.9
        schema = cls._removesuffix(path.name, ".txt")

        with Path(input_file).open("r") as fp:
            lines = fp.read()

        return cls.parse_dialogue(lines, schema)

    @classmethod
    def parse_dialogue(cls, text: str, schema: str) -> dict:
        messages = []
        system = None

        lines = text.split("<|end|>")
        user_message = None

        for line_in in lines:
            line = line_in.strip()

            if len(line) == 0:
                continue

            match = cls.PARSER.match(line)
            if match is None:
                msg = f"Could not find starting token in: {line}"  # type: ignore[no-redef]
                raise ValueError(msg)

            groupdict = match.groupdict()
            role = groupdict["role"]

            if role == "system":
                if system is not None:
                    msg = f"Only one system token allowed in: {line}"
                    raise ValueError(msg)
                system = groupdict["content"]
            elif role == "user":
                if user_message is not None:
                    msg: str = f"Found user token without assistant token: {line}"  # type: ignore[no-redef]
                    raise ValueError(msg)
                user_message = groupdict
            elif role == "assistant":
                if user_message is None:
                    msg = "Found assistant token without user token: {line}"
                    raise ValueError(msg)
                messages.append(user_message)
                messages.append(groupdict)
                user_message = None
            else:
                msg = f"Unknown token: {role}"
                raise ValueError(msg)

        return {"schema": schema, "system": system, "messages": messages}


class ChatKinetica(BaseChatModel):
    """Kinetica LLM Chat Model API.

    Prerequisites for using this API:

    * The ``gpudb`` and ``typeguard`` packages installed.
    * A Kinetica DB instance.
    * Kinetica host specified in ``KINETICA_URL``
    * Kinetica login specified ``KINETICA_USER``, and ``KINETICA_PASSWD``.
    * An LLM context that specifies the tables and samples to use for inferencing.

    This API is intended to interact with the Kinetica SqlAssist LLM that supports
    generation of SQL from natural language.

    In the Kinetica LLM workflow you create an LLM context in the database that provides
    information needed for infefencing that includes tables, annotations, rules, and
    samples. Invoking ``load_messages_from_context()`` will retrieve the contxt
    information from the database so that it can be used to create a chat prompt.

    The chat prompt consists of a ``SystemMessage`` and pairs of
    ``HumanMessage``/``AIMessage`` that contain the samples which are question/SQL
    pairs. You can append pairs samples to this list but it is not intended to
    facilitate a typical natural language conversation.

    When you create a chain from the chat prompt and execute it, the Kinetica LLM will
    generate SQL from the input. Optionally you can use ``KineticaSqlOutputParser`` to
    execute the SQL and return the result as a dataframe.

    The following example creates an LLM using the environment variables for the
    Kinetica connection. This will fail if the API is unable to connect to the database.

    Example:
        .. code-block:: python

            from langchain_kinetica import ChatKinetica

            kinetica_llm = ChatKinetica()

    If you prefer to pass connection information directly then you can create a
    connection using ``GPUdb.get_connection()``.

    Example:
        .. code-block:: python

            from langchain_kinetica import ChatKinetica

            kdbc = GPUdb.get_connection()
            kinetica_llm = ChatKinetica(kdbc=kdbc)
    """

    kdbc: GPUdb | None = Field(exclude=True)
    """ Kinetica DB connection. """

    @pre_init
    def validate_environment(cls, values: dict) -> dict:  # noqa: N805
        """Pydantic object validator."""
        kdbc = values.get("kdbc")
        if kdbc is None:
            kdbc = GPUdb.get_connection()
            values["kdbc"] = kdbc
        return values

    @property
    def _llm_type(self) -> str:
        return "kinetica-sqlassist"

    def _get_kdbc(self) -> GPUdb:
        if self.kdbc is None:
            msg = "Kinetica DB connection is not initialized."
            raise ValueError(msg)
        return self.kdbc

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "kinetica_version": str(self._get_kdbc().server_version),
            "api_version": version("gpudb"),
        }

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop is not None:
            msg = "stop kwargs are not permitted."
            raise ValueError(msg)

        dict_messages = [self._convert_message_to_dict(m) for m in messages]
        sql_response = self._submit_completion(dict_messages)

        response_message = cast("_KdtMessage", sql_response.choices[0].message)
        generated_dict = response_message.model_dump()

        generated_message = self._convert_message_from_dict(generated_dict)

        llm_output = {
            "input_tokens": sql_response.usage.prompt_tokens,
            "output_tokens": sql_response.usage.completion_tokens,
            "model_name": sql_response.model,
        }
        return ChatResult(
            generations=[ChatGeneration(message=generated_message)],
            llm_output=llm_output,
        )

    def load_messages_from_context(self, context_name: str) -> list:
        """Load a lanchain prompt from a Kinetica context.

        A Kinetica Context is an object created with the Kinetica Workbench UI or with
        SQL syntax. This function will convert the data in the context to a list of
        messages that can be used as a prompt. The messages will contain a
        ``SystemMessage`` followed by pairs of ``HumanMessage``/``AIMessage`` that
        contain the samples.

        Args:
            context_name: The name of an LLM context in the database.

        Returns:
            A list of messages containing the information from the context.
        """
        # query kinetica for the prompt
        sql = f"GENERATE PROMPT WITH OPTIONS (CONTEXT_NAMES = '{context_name}')"

        result = self._execute_sql(sql)
        prompt = result["Prompt"]
        prompt_json = json.loads(prompt)

        # convert the prompt to messages
        request = _KdtoSuggestRequest.model_validate(prompt_json)
        payload = request.payload

        dict_messages = []
        dict_messages.append({"role": "system", "content": payload.get_system_str()})

        dict_messages.extend(payload.get_messages())
        return [self._convert_message_from_dict(m) for m in dict_messages]

    def _submit_completion(self, messages: list[dict]) -> _KdtSqlResponse:
        """Submit a /chat/completions request to Kinetica."""
        request = {"messages": messages}
        request_json = json.dumps(request)
        response_raw = self._get_kdbc()._GPUdb__submit_request_json(  # noqa: SLF001
            "/chat/completions", request_json
        )
        response_json = json.loads(response_raw)

        status = response_json["status"]
        if status != "OK":
            message = response_json["message"]
            match_resp = re.compile(r"response:({.*})")
            result = match_resp.search(message)
            if result is not None:
                response = result.group(1)
                response_json = json.loads(response)
                message = response_json["message"]
            raise ValueError(message)

        data = response_json["data"]
        response = _KdtCompletionResponse.model_validate(data)
        if response.status != "OK":
            msg = "SQL Generation failed."
            raise ValueError(msg)
        return response.data

    def _execute_sql(self, sql: str) -> dict:
        """Execute an SQL query and return the result."""
        response = self._get_kdbc().execute_sql_and_decode(
            sql, limit=1, get_column_major=False
        )

        status_info = response["status_info"]
        if status_info["status"] != "OK":
            message = status_info["message"]
            raise ValueError(message)

        records = response["records"]
        if len(records) != 1:
            msg = "No records returned."
            raise ValueError(msg)

        record = records[0]
        return dict(record)

    @classmethod
    def load_messages_from_datafile(cls, sa_datafile: Path) -> list[BaseMessage]:
        """Load a lanchain prompt from a Kinetica context datafile."""
        datafile_dict = _KineticaLlmFileContextParser.parse_dialogue_file(sa_datafile)
        return cls._convert_dict_to_messages(datafile_dict)

    @classmethod
    def _convert_message_to_dict(cls, message: BaseMessage) -> dict:
        """Convert a single message to a BaseMessage."""
        content = cast("str", message.content)
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            msg = f"Got unsupported message type: {message}"
            raise TypeError(msg)

        return {"role": role, "content": content}

    @classmethod
    def _convert_message_from_dict(cls, message: dict) -> BaseMessage:
        """Convert a single message from a BaseMessage."""
        role = message["role"]
        content = message["content"]
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        if role == "system":
            return SystemMessage(content=content)
        msg = f"Got unsupported role: {role}"
        raise ValueError(msg)

    @classmethod
    def _convert_dict_to_messages(cls, sa_data: dict) -> list[BaseMessage]:
        """Convert a dict to a list of BaseMessages."""
        schema = sa_data["schema"]
        system = sa_data["system"]
        messages = sa_data["messages"]
        LOG.info("Importing prompt for schema: %s", schema)

        result_list: list[BaseMessage] = []
        result_list.append(SystemMessage(content=system))
        result_list.extend([cls._convert_message_from_dict(m) for m in messages])
        return result_list


class KineticaSqlResponse(BaseModel):
    """Response containing SQL and the fetched data.

    This object is returned by a chain with ``KineticaSqlOutputParser`` and it contains
    the generated SQL and related Pandas Dataframe fetched from the database.
    """

    sql: str = Field(default="")
    """The generated SQL."""

    dataframe: Any = Field(default=None)
    """The Pandas dataframe containing the fetched data."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class KineticaSqlOutputParser(BaseOutputParser[KineticaSqlResponse]):
    """Fetch and return data from the Kinetica LLM.

    This object is used as the last element of a chain to execute generated SQL and it
    will output a ``KineticaSqlResponse`` containing the SQL and a pandas dataframe with
    the fetched data.

    Example:
        .. code-block:: python

            from langchain_kinetica import ChatKinetica, KineticaSqlOutputParser

            kinetica_llm = ChatKinetica()

            # create chain
            ctx_messages = kinetica_llm.load_messages_from_context(self.context_name)
            ctx_messages.append(("human", "{input}"))
            prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
            chain = (
                prompt_template
                | kinetica_llm
                | KineticaSqlOutputParser(kdbc=kinetica_llm.kdbc)
            )
            sql_response: KineticaSqlResponse = chain.invoke(
                {"input": "What are the female users ordered by username?"}
            )

            assert isinstance(sql_response, KineticaSqlResponse)
            LOG.info(f"SQL Response: {sql_response.sql}")
            assert isinstance(sql_response.dataframe, pd.DataFrame)
    """

    kdbc: Any = Field(exclude=True)
    """ Kinetica DB connection. """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @override
    def parse(self, text: str) -> KineticaSqlResponse:
        """Parse the LLM output text to fetch data from Kinetica."""
        df = self.kdbc.to_df(text)
        return KineticaSqlResponse(sql=text, dataframe=df)

    @override
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> KineticaSqlResponse:
        """Parse the LLM output result to fetch data from Kinetica."""
        return self.parse(result[0].text)

    @property
    def _type(self) -> str:
        return "kinetica_sql_output_parser"
