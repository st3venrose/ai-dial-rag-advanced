import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'

# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(
        self,
        inputs: list[str],
        dimensions: int = 1536,
        print_request: bool = False,
        **kwargs,
    ) -> dict[int, list[float]]:
        """
        Generate embeddings for the input strings using the DIAL embeddings API.
        Args:
            inputs (list[str]): List of input strings to embed.
            dimensions (int): Number of dimensions for the embeddings.
            print_request (bool): Whether to print the request details.
            **kwargs: Additional keyword arguments to pass to the API request.
        Returns:
            dict[int, list[float]]: Dictionary with indexed embeddings (key is index from input list and value vector list).
        """
        if not inputs:
            raise ValueError("`inputs` must be a non-empty list of strings")

        if print_request:
            print(f"Getting embeddings for {len(inputs)} inputs with dimensions={dimensions}")

        response = self.fetch_embeddings(inputs, dimensions, **kwargs)
        items = response.get("data", [])
        indexed_embeddings: dict[int, list[float]] = {}

        if not items:
            raise ValueError("No embeddings data present in the response")

        for item in items:
            index = item.get("index")
            embedding = item.get("embedding")
            if index is None or embedding is None:
                continue
            indexed_embeddings[int(index)] = embedding

        if not indexed_embeddings:
            raise ValueError("Embeddings data could not be parsed from the response")

        return indexed_embeddings

    def fetch_embeddings(self, inputs: list[str], dimensions: int = 1536, **kwargs) -> requests.Response:
        """
        Prepare and send a request to the DIAL embeddings API to generate embeddings for the input strings.
        Args:
            inputs (list[str]): List of input strings to embed.
            dimensions (int): Number of dimensions for the embeddings.
            **kwargs: Additional keyword arguments to pass to the API request.

        Returns:
            requests.Response: Response object containing the API response.
        """
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }

        request_data = {
            "input": inputs,
            "dimensions": dimensions,
            **kwargs,
        }

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=request_data,
            timeout=60,
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        return response.json()

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
