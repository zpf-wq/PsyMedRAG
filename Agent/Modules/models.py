from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv()


class Router:
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant datasource."""

        datasource: Literal["vectorstore", "casual_convo", "mental_query"] = Field(
            ...,
            description="Given a user question choose to route it to casual_convo or a vectorstore.",
        )

    # LLM with function call
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """
    You are an expert at routing a user query to the correct datasource.
    Choose:
    - "mental_query" if the question is about psychology, emotions, mental health, or counseling.
    - "casual_convo" if the question is just small talk, daily chat, or unrelated to psychology/medicine.
    - "vectorstore" if the question is about medicine, pharmacology, biology, or requires medical/academic information.
    
    For example:
        user: I feel anxious these days, what should I do? → mental_query
        user: Hey, how's the weather today? → casual_convo
        user: What are the side effects of ibuprofen? → vectorstore
        user: Can you explain how dopamine works in the brain? → vectorstore
        user: I'm lonely, can you chat with me? → mental_query
"""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    @staticmethod
    def get_model():
        return Router.question_router


class DocGrader:
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return DocGrader.retrieval_grader


class Generator:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialised assistant for question-answering tasks. 
                         Use the following pieces of retrieved context and message history to answer the question.
                         If the answer is not provided in the retrieved documents and message history, just say that you don't know. 
                         Keep the answer detailed and clear."""),
        ("human", """Use the following pieces of retrieved context and message history to answer the question.
                         If the answer is not provided in the retrieved documents and message history, just say that you don't know. 
                         Only use the context to answer the question.
                            Question: {question}        
                            Context: {context} 
                            Answer:"""),
    ])

    # LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    @staticmethod
    def get_model():
        return Generator.rag_chain


class HallucinationGrader:
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""

        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. 
        If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'. Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return HallucinationGrader.hallucination_grader


class AnswerGrader:
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""

        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
        If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "If the LLM Generation is saying that it doesnt know or not sure or stating to keep the questions relevant to topic , grade it as 'yes'. User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    @staticmethod
    def get_model():
        return AnswerGrader.answer_grader


class QuestionRewriter:
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-893dde948ece417a94b24d5c7e56a802"
    )

    system = """You are a helper that preserves the user's question exactly as given.
    Do not change or optimize the question. Just return it as is.
    """
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Here is the user question: \n\n {question} \n\n CHAT HISTORY : {chat_history}. Return the question unchanged."),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    @staticmethod
    def get_model():
        return QuestionRewriter.question_rewriter

# rewriter = QuestionRewriter.get_model()
#
# result = rewriter.invoke({
#     "question": "它是不是也会有偏见？",
#     "chat_history": "我们刚才在聊白血病的危害，比如偏见和错误信息。"
# })
#
# print(result)




