# container agent

Purpose is to explore how a simple reasoning, search, shell
agent interacts with and operates on a container.

- has a server that runs on a container
- has a client that runs on your local
- has internet search tool, shell tool, specific web page search tool

### setup

in `container-agent` run `touch .env`

here is an example env

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
TAVILY_API_KEY=
```

The agent uses the `OPENAI_API_KEY` for LLM and `TAVILY_API_KEY`
for search by default but why not get all intelligence providers?

### running

There are two components

- server
- client

With Docker running, in the `container-agent` directory,
run `docker compose up --build -d`.
The server will automatically run on that container.

To run the client,
`cd client` then

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 client.py
```

The client will be connected to the container agent.
Send `hello` to verify.
