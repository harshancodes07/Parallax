"""
Thin, provider-agnostic LLM client.

This is the ONLY file that should change when the team confirms which
LLM provider we're using (OpenAI, Anthropic, etc.). Nothing else in
teachback/ should need to know or care which provider is behind this.

Contract:
    await call_llm(prompt: str) -> str

Must return the raw text of the model's response. The caller
(evaluator.py) is responsible for parsing/validating that text as JSON.
Do not add retry/fallback logic here that silently changes behaviour —
teachback uses strict parsing upstream, so this function should raise
on genuine failures (network errors, auth errors) rather than swallow them.
"""


async def call_llm(prompt: str) -> str:
    """
    Send `prompt` to the shared LLM provider and return its raw text response.

    NOT YET IMPLEMENTED: waiting on team decision for shared LLM provider/API key.
    Wire this up once confirmed, e.g.:

        client = SomeProviderClient(api_key=os.environ["LLM_API_KEY"])
        response = await client.complete(prompt)
        return response.text
    """
    raise NotImplementedError(
        "LLM provider not yet wired. Set this up once the team confirms "
        "the shared provider/API key, then update only this function."
    )
