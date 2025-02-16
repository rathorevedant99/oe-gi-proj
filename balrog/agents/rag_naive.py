import copy
import re

from balrog.agents.base import BaseAgent
from balrog.agents.agent_rag_utils import *


class RAGNaiveAgent(BaseAgent):
    """An agent that generates actions guided by RAG based on observations without complex reasoning."""

    def __init__(self, client_factory, prompt_builder, config):
        """Initialize the RAGNaiveAgent with a client and prompt builder."""
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.faiss_index = config.faiss_index_path
        self.page_titles = config.page_titles_path
        self.embedding_model = config.embedding_model
        self.top_k = config.top_k

    def act(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        query = obs["text"]["short_term_context"]  # Use short-term context as the query to FAISS
        retrieved_docs = search_faiss(self.faiss_index, self.page_titles, query, model=self.embedding_model, top_k=self.top_k)






        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction

        response = self.client.generate(messages)

        final_answer = self._extract_final_answer(response)

        return final_answer
    

    def _extract_final_answer(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = final_answer._replace(completion=filter_letters(final_answer.completion))

        return final_answer
