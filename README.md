# langchain-kinetica

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-kinetica?label=%20)](https://pypi.org/project/langchain-kinetica/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-kinetica)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-kinetica)](https://pypistats.org/packages/langchain-kinetica)
[![Test](https://github.com/kineticadb/langchain-kinetica/actions/workflows/test.yml/badge.svg)](https://github.com/kineticadb/langchain-kinetica/actions/workflows/test.yml)

This project contains the Kinetica integration 
provider for Langchain. [Kinetica](https://www.kinetica.com/) is a real-time database purpose built for enabling
analytics and generative AI on time-series & spatial data. 

- [Features](#features)
- [Quick Install](#quick-install)
- [Documentation](#documentation)
- [Testing](#testing)
- [See Also](#see-also)

## Features

This package provides integration for core capabilities:

- **Chat model** - Kinetica native Text-to-SQL Generation.
- **Vector Store** - Vector similarity search using Kinetica tables.
- **Document Loader** - Generate embeddings from Kinetica tables.

For more information see the 
[Kinteica Provider Docs](https://docs.langchain.com/oss/python/integrations/providers/kinetica)

## Quick Install

```bash
pip install langchain-deepseek
```

## Documentation

For conceptual guides, tutorials, and examples on using these classes, see the [Kinteica Provider Docs](https://docs.langchain.com/oss/python/integrations/providers/deepseek).

The documentation is also available in notebook format under `./notebooks`.

## Testing

In addition the `make test` macro these tests can be run individually.

```
$ make integration_tests TEST_FILE=tests/integration_tests/test_chat_models.py

$ make integration_tests TEST_FILE=tests/integration_tests/test_vectorstores.py::test_kinetica
```

## See Also

- [Kinetica LLM Documentation](https://docs.kinetica.com/7.2/sql-gpt/)
- [LangChain Chat Models](https://docs.langchain.com/oss/python/langchain/models)
- [Contributing to LangChain](https://docs.langchain.com/oss/python/contributing/overview)
