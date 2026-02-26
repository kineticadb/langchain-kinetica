"""Test Kinetica Chat API wrapper."""

import logging
from collections.abc import Generator

import faker
import pandas as pd
import pytest
from gpudb import GPUdb, GPUdbTable
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from langchain_kinetica import (
    ChatKinetica,
    KineticaSqlOutputParser,
    KineticaSqlResponse,
)

LOG = logging.getLogger(__name__)

# Schema to be created for tests
SCHEMA_NAME = "test"


@pytest.fixture(scope="module")
def chat_kinetica() -> ChatKinetica:
    """Fixture to create a `ChatKinetica` instance for the tests."""
    kdbc = GPUdb.get_connection()
    return ChatKinetica(kdbc=kdbc)  # type: ignore[call-arg]


class TestChatKinetica:
    """Integration tests for `Kinetica` chat models.

    You will need to configure connection parameters:
    ```
        export KINETICA_HOST="http://localhost:9191"
        export KINETICA_USERNAME="demo"
        export KINETICA_PASSWORD="xxx"
    ```

    For more information see https://docs.kinetica.com/7.1/sql-gpt/concepts/.

    These integration tests follow a workflow:

    1. The `test_setup()` will create a table with fake user profiles and and a related
       LLM context for the table.

    2. The LLM context is retrieved from the DB and used to create a chat prompt
       template.

    3. A chain is constructed from the chat prompt template.

    4. The chain is executed to generate the SQL and execute the query.
    """

    table_name = f"{SCHEMA_NAME}.test_profiles"
    context_name = f"{SCHEMA_NAME}.test_llm_ctx"
    num_records = 100

    # @pytest.mark.vcr()
    def test_setup(self, chat_kinetica: ChatKinetica) -> None:
        """Create the connection, test table, and LLM context."""
        self._create_test_table(
            kinetica_dbc=chat_kinetica.kdbc,
            table_name=self.table_name,
            num_records=self.num_records,
        )
        self._create_llm_context(
            kinetica_dbc=chat_kinetica.kdbc, context_name=self.context_name
        )

    # @pytest.mark.vcr()
    def test_create_llm(self, chat_kinetica: ChatKinetica) -> None:
        """Create an LLM instance."""
        LOG.info(chat_kinetica._identifying_params)

        assert isinstance(chat_kinetica.kdbc, GPUdb)
        assert chat_kinetica._llm_type == "kinetica-sqlassist"

    # @pytest.mark.vcr()
    def test_load_context(self, chat_kinetica: ChatKinetica) -> None:
        """Load the LLM context from the DB."""
        ctx_messages = chat_kinetica.load_messages_from_context(self.context_name)

        system_message = ctx_messages[0]
        assert isinstance(system_message, SystemMessage)

        last_question = ctx_messages[-2]
        assert isinstance(last_question, HumanMessage)
        assert last_question.content == "How many male users are there?"

    # @pytest.mark.vcr()
    def test_generate(self, chat_kinetica: ChatKinetica) -> None:
        """Generate SQL from a chain."""
        # create chain
        ctx_messages = chat_kinetica.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = prompt_template | chat_kinetica

        resp_message = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )
        LOG.info("SQL Response: %s", resp_message.content)
        assert isinstance(resp_message, AIMessage)

    # @pytest.mark.vcr()
    def test_full_chain(self, chat_kinetica: ChatKinetica) -> None:
        """Generate SQL from a chain and execute the query."""
        # create chain
        ctx_messages = chat_kinetica.load_messages_from_context(self.context_name)
        ctx_messages.append(("human", "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(ctx_messages)
        chain = (
            prompt_template
            | chat_kinetica
            | KineticaSqlOutputParser(kdbc=chat_kinetica.kdbc)
        )
        sql_response: KineticaSqlResponse = chain.invoke(
            {"input": "What are the female users ordered by username?"}
        )

        assert isinstance(sql_response, KineticaSqlResponse)
        LOG.info("SQL Response: %s", sql_response.sql)
        assert isinstance(sql_response.dataframe, pd.DataFrame)
        users = sql_response.dataframe["username"]
        assert users[0] == "alexander40"

    @classmethod
    def _create_fake_records(cls, count: int) -> Generator:
        """Generator for fake records."""
        faker.Faker.seed(5467)
        faker_inst = faker.Faker(locale="en-US")
        for rec_id in range(count):
            rec = dict(id=rec_id, **faker_inst.simple_profile())
            rec["birthdate"] = pd.Timestamp(rec["birthdate"])
            yield rec

    @classmethod
    def _create_test_table(
        cls, kinetica_dbc: GPUdb, table_name: str, num_records: int
    ) -> GPUdbTable:
        """Create a table from the fake records generator."""
        table_df = pd.DataFrame.from_records(
            data=cls._create_fake_records(num_records), index="id"
        )

        kinetica_dbc.create_schema(
            schema_name=SCHEMA_NAME,
            options={"no_error_if_exists": "true"}
        )
        #cls._check_error(response)

        LOG.info("Creating test table '%s' with %d records...", table_name, num_records)
        return GPUdbTable.from_df(
            df=table_df,
            db=kinetica_dbc,
            table_name=table_name,
            clear_table=True,
            load_data=True,
            column_types={},
        )

    @classmethod
    def _check_error(cls, response: dict) -> None:
        """Convert a DB error into an exception."""
        status = response["status_info"]["status"]
        if status != "OK":
            resp_message = response["status_info"]["message"]
            msg = f"[{status}]: {resp_message}"
            raise ValueError(msg)

    @classmethod
    def _create_llm_context(cls, kinetica_dbc: GPUdb, context_name: str) -> None:
        """Create an LLM context for the table."""
        sql = f"""
        CREATE OR REPLACE CONTEXT {context_name}
        (
            TABLE = {cls.table_name}
            COMMENT = 'Contains user profiles.'
        ),
        (
            SAMPLES = (
            'How many male users are there?' =
            'select count(1) as num_users
            from {cls.table_name}
            where sex = ''M'';')
        )
        """  # noqa: S608
        LOG.info("Creating context: %s", context_name)
        response = kinetica_dbc.execute_sql(sql)
        cls._check_error(response)
