# langchain-kinetica example notebooks

These notebooks are used to generate documentation in the 
[langchain-docs](https://github.com/langchain-ai/docs) repository and will be 
published to the official [langchain docs](https://docs.langchain.com/oss/python/integrations/providers/kinetica).

To update the documentation:

1. Make changes to the notebooks and save the changes with the results.

2. run `make docs` to generate the markdown files in `./md_docs`:
    ```
    ./md_docs/kinetica_provider.md
    ./md_docs/kinetica_vectorstore.md
    ./md_docs/kinetica_loader.md
    ./md_docs/kinetica_chat.md
    ./md_docs/kinetica_retriever.md
    ```

3. Clone the kinetica fork of [langchain-docs](https://github.com/kineticadb/langchain-docs)

4. Checkout a new a feature branch in `langchain-docs`.

5. Diff the markdown files with the associated files in `langchain-docs` and merge changes.
    ```
    ./src/oss/python/integrations/retrievers/kinetica.mdx
    ./src/oss/python/integrations/chat/kinetica.mdx
    ./src/oss/python/integrations/providers/kinetica.mdx
    ./src/oss/python/integrations/vectorstores/kinetica.mdx
    ./src/oss/python/integrations/document_loaders/kinetica.mdx
    ```

6. Follow the LangChain 
   [contribution guidelines](https://docs.langchain.com/oss/python/contributing/documentation).
   (e.g. run `format` and `lint`)

7. Commit changes and open a PR to merge the Kinetica fork to the LanChain repo.
