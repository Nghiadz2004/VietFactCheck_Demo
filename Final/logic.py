from dotenv import load_dotenv
import os
import asyncio
from openai import OpenAI
from Module.module_1_2 import Module_1_2
from Module.module_3 import Module3
from Module.module_4 import Module_4
from Module.module_5 import Module_5

# Lấy API key từ biến môi trường
load_dotenv(".env") # load biến môi trường từ file .env
openai_api_key = os.environ["GPT_TOKEN"]
ming3993_token = os.environ["MING3993_TOKEN"]
SERPER_API_KEY = os.environ["SERPER_API_KEY"]
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["JAVA_HOME"], "bin")

client = OpenAI(api_key=openai_api_key)
detector = Module_1_2()
module_3 = Module3(
    openai_api_key=openai_api_key,
    serper_api_key=SERPER_API_KEY
)
input_file = 'document_retrieval_results.json'
module_4_results = None
module_4 = Module_4()
verdict_engine = Module_5(client)

import os

os.chdir("..")
print(os.getcwd())



async def Fact_Checking_Pipeline(input_text: str):

    # Module 1 + 2: Làm sạch và trích xuất các câu claim
    CLEANED_CLAIMS = detector.detect_claims(input_text=input_text, client=client)

    if not CLEANED_CLAIMS:
        return

    # 2. Chạy Module 3: Retrieval
    retrieval_data = module_3.run(claims=CLEANED_CLAIMS, output='document_retrieval_results.json')


    # Để chuyển sang format input của Module 4 (Dict[str, List[Dict]]), cần convert:
    input_data_m4 = {item["claim"]: item["sources"] for item in retrieval_data}

    results_with_stance = await module_4.run(input_data_m4)

    if not results_with_stance:
        return

    # Gọi Module 5 với output của Module 4
    final_results = verdict_engine.run(results_with_stance)
    print(final_results)
    return final_results
