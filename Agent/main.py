from dotenv import load_dotenv
from Modules.models import Router, DocGrader, Generator, HallucinationGrader,AnswerGrader,QuestionRewriter
from Modules.node import Nodes
from Modules.graph import Graph
from Retriever.retriever import Retriever

load_dotenv()


class NodeHelpers:
    retriever = Retriever()

    question_router = Router.get_model()

    retrieval_grader = DocGrader.get_model()

    rag_chain = Generator.get_model()

    hallucination_grader = HallucinationGrader.get_model()

    answer_grader = AnswerGrader.get_model()

    question_rewriter = QuestionRewriter.get_model()


nodes = Nodes(NodeHelpers())

app = Graph.create(nodes)

inp = "hello"

message_history = ["the messages till now are given below: \n\n"]
while inp != "exit":
    inp = input("Enter question: ")
    inputs = {"question": inp, "message_history": message_history}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}': \nREGENERATED : {nodes.REGENERATION_COUNT}\nRETREVIAL : {nodes.RERETREVIAL_COUNT}")

        print("\n---\n")

    # Final generation
    print(value["generation"])

    # print(sentence)
    value["message_history"].append(
        "msg number [" + str(len(value["message_history"])) + "] agent : " + value["generation"])
    message_history = value["message_history"]
    # print(message_history)
    for doc in value["documents"]:
        print(doc)
        # print("site : " + cite_mapper[doc.metadata["source"]] + "\t  page : " +str( doc.metadata["page"]) )
