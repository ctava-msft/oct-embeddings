lab guide:
https://github.com/AzureCosmosDB/cosmosdb-nosql-copilot/blob/start/lab/lab-guide.md

git clone https://github.com/AzureCosmosDB/cosmosdb-nosql-copilot

az cosmosdb sql role assignment create --account-name "cosmos-khifsnz7gfujg" --resource-group "cosmodb-lab" --scope "/" --principal-id $(az ad signed-in-user show --query id -o tsv) --role-definition-id "00000000-0000-0000-0000-000000000002"



dotnet workload restore --project ./src/cosmos-copilot.AppHost/cosmos-copilot.AppHost.csproj
dotnet run --project ./src/cosmos-copilot.AppHost/cosmos-copilot.AppHost.csproj


How to load data:
https://github.com/AzureCosmosDB/BRK193-Ignite2024/blob/main/cosmos-search-demo/src/data/data-loader.py

where is the data for this lab?:
https://cosmosdbcosmicworks.blob.core.windows.net/cosmic-works-vectorized/product-text-3-large-1536-llm-gen-2.json

Which embeddings model:
https://github.com/marketplace/models/azure-openai/text-embedding-3-large

Python version is there too:
https://github.com/AzureCosmosDB/quickstart-nosql-python
needs infra and semantic kernel


