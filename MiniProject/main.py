
from langchain.agents import SimpleAgent
from langchain.rl import RLAgent
from langchain.rl import NoReward
from langchain.llms import ChatOpenAI

def main():

    # Create an LLM (Language Model). Here, we'll use OpenAI's model for demonstration.
    llm = ChatOpenAI(openai_api_key="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")



    # Create a simple agent
    agent = SimpleAgent(llm_name='openai')

    # Or create an RL agent
    # rl_agent = RLAgent(llm_name='openai', reward_function=NoReward())

    # Add agent to LangChain
    lc.add_agent('simple', agent)
    # lc.add_agent('rl', rl_agent)

    # Use the agent
    while True:
        user_input = input("Ask me a question: ")
        if user_input.lower() == 'exit':
            break

        response = lc.use_agent('simple', user_input)
        print(response)

if __name__ == "__main__":
    main()
