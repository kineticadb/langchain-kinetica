# langchain-kinetica

Kinetica intefrace for Langchain. See the [LLM documentation][LLM_DOCS] for an overview of the Kinetica LLM.

[LLM_DOCS]: <https://docs.kinetica.com/7.1/sql-gpt/>

- [1. Prerequisites](#1-prerequisites)
- [2. Package Contents](#2-package-contents)
- [3. Installation](#3-installation)
- [4. Usage](#4-usage)
- [5. Building](#5-building)
- [6. See Also](#6-see-also)

## 1. Prerequisites

To use langchain with Kinetica you will need:

* Python runtime >3.10
* Kinetica SqlAssist LLM
* Kinetica instance >7.2.0 configured to use SqlAssist.

## 2. Package Contents

* `KineticaChatLLM`: ChatModel for converting natural language to SQL.
* `KineticaSqlOutputParser`: OutputParser that will execute SQL from the `KineticaChatLLM`.
* `SqlResponse`: If the Kinetica chain ends with `KineticaSqlOutputParser` then this response will contain the generated SQL and results from its execution.

## 3. Installation

This project is not yet available on pypi. You can install it directly from the repository.

```sh
$ pip install "langchain-kinetica @ git+ssh://git@github.com/kineticadb/langchain-kinetica.git"
```

## 4. Usage

See the [Kinetica LLM Demo notebook](./notebooks/kinetica_llm_demo.ipynb) for examples.

## 5. Building

Install the project locally.

```sh
$ pip install --editable .
```

You will need to install the build utility.

```sh
$ pip install --upgrade build
```

Build the project

```sh
$ python3 -m build
```

The build will generate a `.whl` file that can be distributed.

```sh
$ ls -1 ./dist
langchain-kinetica-1.0.tar.gz
langchain_kinetica-1.0-py3-none-any.whl
```

## 6. See Also

- [Kinetica LLM Documentation](https://docs.kinetica.com/7.1/sql-gpt/)
- [LangChain Prompts](https://python.langchain.com/docs/modules/model_io/prompts/)
- [LancChain Chat Models](https://python.langchain.com/docs/modules/model_io/chat/)
