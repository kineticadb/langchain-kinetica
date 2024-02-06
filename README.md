# langchain-kinetica

Kinetica intefrace for Langchain. See the [LLM documentation][LLM_DOCS] for an overview of the Kinetica LLM.

[LLM_DOCS]: <https://docs.kinetica.com/7.1/sql-gpt/>

## Prerequisites

To use langchain with Kinetica you will need:

* Python runtime >3.10
* Kinetica SqlAssist LLM
* Kinetica instance >7.2.0 configured to use SqlAssist.

## Package Contents

* `KineticaChatLLM`: ChatModel for converting natural language to SQL.
* `KineticaSqlOutputParser`: OutputParser that will execute SQL from the `KineticaChatLLM` and return a `SqlResponse`
* `SqlResponse`: The output of the Kinetica chain containing the generated SQL and results from its execution.

## Installation

This project is not yet available on pypi. You can install it directly from the repository.

```sh
$ pip install "langchain-kinetica @ git+ssh://git@github.com/kineticadb/langchain-kinetica.git"
```

## Usage

See the [Kinetica LLM Demo notebook](./notebooks/kinetica_llm_demo.ipynb) for examples.

## Building

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
