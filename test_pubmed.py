from langchain_community.utilities import PubMedAPIWrapper

print("Testing PubMed API Directly...")
try:
    # This bypasses your agent_tools.py and tests LangChain directly
    api_wrapper = PubMedAPIWrapper()
    result = api_wrapper.run("latest treatments for glioblastoma")
    print("\n--- RESULTS ---")
    print(result)
except Exception as e:
    print(f"\n❌ PubMed Tool Failed: {e}")