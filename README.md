# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
```
GenerativeCategorization
├─ .devcontainer
│  ├─ Dockerfile
│  ├─ add-notice.sh
│  ├─ devcontainer.json
│  └─ noop.txt
├─ .streamlit
│  └─ config.toml
├─ CategoryEvaluator
│  ├─ confidence_intervals.py
│  ├─ metrics_report.md
│  └─ workers
│     ├─ __init__.py
│     ├─ class_balance_checker.py
│     ├─ data_loader.py
│     ├─ evaluator.py
│     ├─ main.py
│     ├─ merge_handler.py
│     └─ mergers
│        ├─ __init__.py
│        ├─ base_merger.py
│        ├─ exact_merger.py
│        └─ fuzzy_merger.py
├─ Docker
│  └─ startup.sh
├─ LabeLMaker
│  ├─ Categorizer.py
│  ├─ Fewshot.py
│  ├─ Manyshot.py
│  ├─ Zeroshot.py
│  ├─ __init__.py
│  └─ utils
│     ├─ __init__.py
│     ├─ categorize_handler.py
│     ├─ category.py
│     ├─ dedup_RyanC.py
│     ├─ dedup_conflicts.py
│     ├─ file_manager.py
│     ├─ normalize_text.py
│     └─ page_renderer.py
├─ LabeLMaker_config
│  ├─ __init__.py
│  └─ config.py
├─ Makefile
├─ PlanForWebInterface.md
├─ README.md
├─ Tests
│  ├─ CA_Grants
│  │  ├─ fewshot_test_ca_data.py
│  │  ├─ fewshot_test_ca_script.py
│  │  ├─ zeroshot_test_ca_data.py
│  │  └─ zeroshot_test_ca_script.py
│  ├─ COVIDSentiments
│  │  ├─ EvalCovidResult_FS.py
│  │  ├─ EvalCovidResults_FS.ipynb
│  │  ├─ EvalCovidResults_MS.ipynb
│  │  ├─ EvalCovidResults_ZS.ipynb
│  │  ├─ Merge_FS.py
│  │  ├─ Merge_MS.py
│  │  └─ Merge_ZS.py
│  ├─ DifficultAirways
│  │  └─ concatenate_report_strings.py
│  ├─ MedicalTextClassification
│  │  ├─ EvalMedTextResults_FS.ipynb
│  │  ├─ EvalMedTextResults_ZS.ipynb
│  │  ├─ Merge_FS.py
│  │  └─ Merge_ZS.py
│  ├─ PerformanceComparisons
│  │  └─ CA_Research
│  │     ├─ EvalCAResults_FS.ipynb
│  │     └─ EvalCAResults_ZS.ipynb
│  ├─ __init__.py
│  ├─ fewshot_test_data.py
│  ├─ fewshot_test_script.py
│  ├─ zeroshot_test_data.py
│  └─ zeroshot_test_script.py
├─ app
│  ├─ __init__.py
│  └─ v01
│     ├─ __init__.py
│     └─ schemas.py
├─ assets
│  ├─ gencat_fewshot.prompty
│  └─ gencat_zeroshot.prompty
├─ docs
│  ├─ LabeLMaker
│  │  ├─ data.md
│  │  ├─ generate.md
│  │  ├─ main.md
│  │  ├─ prompts.md
│  │  └─ utils.md
│  ├─ dev_requirements.txt
│  ├─ diagrams
│  │  └─ plantUML
│  │     └─ initialDesign.wsd
│  ├─ index.md
│  ├─ run_aider.sh
│  ├─ serve_docs.sh
│  └─ src_setup.sh
├─ explorations
│  ├─ Convert_Sentiment_Categories.py
│  ├─ just_ada_streamlit.py
│  └─ prettyConfMats.py
├─ llm_utils
│  ├─ .devcontainer
│  │  ├─ Dockerfile
│  │  ├─ add-notice.sh
│  │  ├─ devcontainer.json
│  │  └─ noop.txt
│  ├─ .streamlit
│  │  └─ config.toml
│  ├─ Docker
│  │  └─ startup.sh
│  ├─ Makefile
│  ├─ README.md
│  ├─ __init__.py
│  ├─ aiweb_common
│  │  ├─ ObjectFactory.py
│  │  ├─ UML
│  │  │  ├─ base_classes.png
│  │  │  ├─ classes.png
│  │  │  └─ packages.png
│  │  ├─ WorkflowHandler.py
│  │  ├─ __init__.py
│  │  ├─ fastapi
│  │  │  ├─ __init__.py
│  │  │  ├─ helper_apis.py
│  │  │  └─ schemas.py
│  │  ├─ file_operations
│  │  │  ├─ DocxCreator.py
│  │  │  ├─ UploadManager.py
│  │  │  ├─ __init__.py
│  │  │  ├─ file_handling.py
│  │  │  └─ text_format.py
│  │  ├─ generate
│  │  │  ├─ AugmentedResponse.py
│  │  │  ├─ AugmentedServicer.py
│  │  │  ├─ ChatResponse.py
│  │  │  ├─ ChatSchemas.py
│  │  │  ├─ ChatServicer.py
│  │  │  ├─ PromptAssembler.py
│  │  │  ├─ PromptyResponse.py
│  │  │  ├─ PromptyServicer.py
│  │  │  ├─ QueryInterface.py
│  │  │  ├─ Response.py
│  │  │  ├─ SingleResponse.py
│  │  │  ├─ SingleResponseServicer.py
│  │  │  └─ __init__.py
│  │  ├─ resource
│  │  │  ├─ NIHRePORTERInterface.py
│  │  │  ├─ PubMedInterface.py
│  │  │  ├─ PubMedQuery.py
│  │  │  ├─ __init__.py
│  │  │  └─ default_resource_config.py
│  │  └─ streamlit
│  │     ├─ BYOKLogin.py
│  │     ├─ __init__.py
│  │     └─ streamlit_common.py
│  ├─ docker-compose.yml
│  ├─ docs
│  │  └─ run_aider.sh
│  ├─ requirements.txt
│  ├─ setup.py
│  ├─ test_output.csv
│  └─ workspace.code-workspace
├─ logs
├─ mkdocs.yml
├─ out
│  └─ docs
│     └─ diagrams
│        └─ plantUML
│           └─ initialDesign
│              └─ initialDesign.png
├─ pyproject.toml
├─ requirements.in
├─ requirements.txt
├─ setup.py
├─ user_interface
│  ├─ CategoryEvaluator_app.py
│  ├─ LabeLMaker_app.py
│  ├─ __init__.py
│  └─ mapping_test.py
└─ workspace.code-workspace

```