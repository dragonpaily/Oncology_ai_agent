# In src/agent_tools.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from langchain.tools import tool
import random
import os
from PIL import Image
import nibabel as nib
import google.generativeai as genai
from typing import List
from nilearn.datasets import fetch_atlas_aal
import nilearn.regions # We can import the module directly


# --- Import our organized code modules ---
from .segmentation.model import build_training_model, InstanceNormalization
from .segmentation.utils import Load_nifti, preprocess_sample, analyze_segmentation
from .rag_pipeline import build_retriever
from langchain_community.utilities import PubMedAPIWrapper

print("--- Initializing Agent Tools ---")

# --- 1. SETUP SEGMENTATION MODEL ---
print("🧠 Building and loading segmentation model into memory...")
# IMPORTANT: Make sure you have the model weights file available at this path.
MODEL_WEIGHTS_PATH = 'Data/model weights/Best_model_run_with_derived_metrics_v1.weights.h5' # <-- UPDATE THIS PATH

segmentation_model = None # Define it first
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        with tf.keras.utils.custom_object_scope({'InstanceNormalization': InstanceNormalization}):
            training_model = build_training_model(dropout_rate=0.0)
            training_model.load_weights(MODEL_WEIGHTS_PATH)
            segmentation_model = Model(inputs=training_model.inputs, outputs=training_model.get_layer('out_final').output)
        print("✅ Segmentation model loaded.")
    except Exception as e:
        print(f"❗️ ERROR: Could not load segmentation model weights. Error: {e}")
else:
    print(f"❗️ WARNING: Segmentation model weights not found at {MODEL_WEIGHTS_PATH}. The segmentation tool will not work.")


# --- 2. SETUP RAG RETRIEVER ---
# This builds the vector database from PDFs in the ./data/medical_papers/ folder.
RAG_RETRIEVER = build_retriever(papers_path="Data/Guidelines")

# --- TOOL DEFINITIONS ---

@tool
def run_segmentation_analysis(t1c_path: str, t1n_path: str, t2f_path: str, t2w_path: str) -> dict:
    """
    Analyzes a set of four 3D brain MRI scans (T1c, T1n, FLAIR, T2w) to detect and quantify a tumor.
    Returns a dictionary with tumor volumes in cubic centimeters (cm3) and centroid coordinates in millimeters (mm).
    """
    print(f"🤖 Tool Called: run_segmentation_analysis...")
    
    required_paths = [t1c_path, t1n_path, t2f_path, t2w_path]
    if not all(required_paths):
        return {"error": "Tool failed. All four scan paths (T1c, T1n, T2f, T2w) are required for this analysis."}
    
    
    if segmentation_model is None:
        return {"error": "Segmentation model is not loaded. Cannot perform analysis."}
    try:
        t1c_nii, t1n_data = Load_nifti(t1c_path), Load_nifti(t1n_path).get_fdata()
        t2f_data, t2w_data = Load_nifti(t2f_path).get_fdata(), Load_nifti(t2w_path).get_fdata()
        input_tensor = preprocess_sample(t1c_nii.get_fdata(), t1n_data, t2f_data, t2w_data)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        pred_probs = segmentation_model.predict(input_tensor, verbose=0)
        pred_labels = np.argmax(pred_probs[0], axis=-1)
        analysis_results = analyze_segmentation(pred_labels, t1c_nii.affine)
        print("✅ Tool Finished: Segmentation analysis complete.")
        return analysis_results
    except Exception as e:
        return {"error": f"Could not process NIfTI files. Error: {e}"}

@tool
def calculate_percentage_change(initial_volume: float, final_volume: float) -> float:
    """
    Calculates the percentage change between an initial and final volume. Returns the result as a percentage.
    """
    print(f"🧮 Tool Called: calculate_percentage_change")
    if initial_volume == 0: return 0.0
    change = ((final_volume - initial_volume) / initial_volume) * 100
    print(f"✅ Tool Finished: Calculated change is {change:.2f}%")
    return round(change, 2)

@tool
def clinical_guideline_retriever_tool(query: str) -> str:
    """
    MUST be used to answer any questions that refer to 'the knowledge base', 'guidelines', 'standard-of-care', or 'protocols'.
    This tool searches a local knowledge base of trusted clinical oncology documents. Do not answer these types of questions from memory.
    """
    print(f"📚 Tool Called: clinical_guideline_retriever_tool with query: '{query}'")
    retrieved_docs = RAG_RETRIEVER.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print("✅ Tool Finished: RAG ensemble retrieval complete.")
    return f"Found the following information in the knowledge base:\n{context}"

@tool
def pubmed_search_tool(query: str) -> str:
    """
    Searches the live PubMed database for the absolute latest biomedical literature and clinical trials.
    """
    print(f"🌐 Tool Called: pubmed_search_tool with query: '{query}'")
    api_wrapper = PubMedAPIWrapper()
    result = api_wrapper.run(query)
    print("✅ Tool Finished: PubMed search complete.")
    return result

@tool
def oncokb_query_tool(gene_symbol: str, alteration: str) -> str:
    """
    Queries a simulated OncoKB database to find therapies for a specific genetic alteration (e.g., BRAF V600E).
    """
    print(f"🧬 Tool Called: oncokb_query_tool for {gene_symbol} {alteration}")
    mock_db = {
        "BRAF": {"V600E": "FDA-approved therapies include Dabrafenib and Trametinib for BRAF V600E mutated tumors."},
        "EGFR": {"L858R": "Osimertinib is an FDA-approved therapy for this EGFR mutation in NSCLC."}
    }
    gene_data = mock_db.get(gene_symbol.upper(), {})
    result = gene_data.get(alteration, f"No specific therapies found for {gene_symbol} {alteration}.")
    print("✅ Tool Finished: OncoKB query complete.")
    return result

@tool
def histopathology_mutation_analyzer_tool(histology_image_path: str, gene_to_test: str) -> str:
    """
    Simulates analyzing a histopathology image to predict the mutational status of 'BRAF', 'KRAS', or 'MSI'.
    """
    print(f"🔬 Tool Called: histopathology_mutation_analyzer_tool for {gene_to_test}")
    if gene_to_test.upper() not in ['BRAF', 'KRAS', 'MSI']: return "Error: Invalid gene."
    prob = random.uniform(0.51, 0.99)
    result = f"Simulated analysis of {histology_image_path}: Predicted {gene_to_test.upper()} MUTANT with probability {prob:.2f}."
    print("✅ Tool Finished: Histopathology analysis complete.")
    return result

@tool
def generate_qualitative_report_tool(initial_scan_path: str, followup_scan_path: str, clinical_summary: str) -> str:
    """
    Provides a high-level, qualitative analysis of two brain MRI scans (initial and follow-up) by a generalist vision model.
    Use this tool to get a 'first-look' impression of changes, such as apparent growth or reduction in lesion size.
    This tool does NOT provide precise measurements; for that, use 'run_segmentation_analysis'.
    """
    print("👁️ Tool Called: generate_qualitative_report_tool...")
    try:
        # Helper function to get a 2D slice for the vision model
        def get_2d_slice(nifti_path):
            if not nifti_path or not os.path.exists(nifti_path): return None
            nii = nib.load(nifti_path)
            data = nii.get_fdata()
            axial_slice = data[:, :, data.shape[2] // 2]
            if axial_slice.max() > 0:
                axial_slice = (axial_slice / axial_slice.max()) * 255.0
            return Image.fromarray(np.uint8(axial_slice).T).convert("RGB")

        img_initial = get_2d_slice(initial_scan_path)
        img_final = get_2d_slice(followup_scan_path)

        if img_initial and img_final:
            model_vision = genai.GenerativeModel('gemini-1.5-flash')
            prompt = [
                "You are a radiology assistant. Provide a qualitative analysis of two MRI scans.",
                clinical_summary,
                "Initial Scan:", img_initial,
                "Follow-up Scan:", img_final,
                "Describe any apparent changes and give a one-sentence impression. State your limitations."
            ]
            response = model_vision.generate_content(prompt)
            print("✅ Tool Finished: Qualitative report generated.")
            return response.text
        else:
            return "Could not generate report as one or both scan paths were invalid."
    except Exception as e:
        return f"Error running qualitative vision tool: {e}"




# --- Create a convenient list of all tools for the main app ---
all_tools = [
    run_segmentation_analysis,
    calculate_percentage_change,
    clinical_guideline_retriever_tool,
    pubmed_search_tool,
    oncokb_query_tool,
    histopathology_mutation_analyzer_tool,
    generate_qualitative_report_tool,
]