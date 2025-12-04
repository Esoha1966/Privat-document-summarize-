# Private Document Summarization Tool

## Overview
This Jupyter notebook provides a comprehensive solution for summarizing private documents using Retrieval-Augmented Generation (RAG) with IBM Watsonx AI. The tool loads your private documents, processes them through embeddings, and generates accurate summaries while maintaining document privacy.

## Features
- **Document Processing**: Load and split text documents for efficient processing
- **Vector Storage**: Use ChromaDB for storing document embeddings
- **Secure Summarization**: Process sensitive documents locally with privacy controls
- **IBM Watsonx Integration**: Leverage powerful LLMs through IBM's platform
- **Customizable Parameters**: Control summarization length, creativity, and specificity

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. IBM Watsonx API Access
You need to obtain:
- IBM Cloud API key
- Project ID from IBM Watsonx
- Access to Watsonx AI models

## Installation

### Option 1: Using requirements.txt
Create a `requirements.txt` file with:
```txt
ibm-watsonx-ai==0.2.6
langchain==0.1.16
langchain-ibm==0.1.4
transformers==4.41.2
huggingface-hub==0.23.4
sentence-transformers==2.5.1
chromadb
wget==3.2
torch==2.3.1
```

Then install:
```bash
pip install -r requirements.txt
```

### Option 2: Manual Installation
```bash
pip install ibm-watsonx-ai==0.2.6
pip install langchain==0.1.16
pip install langchain-ibm==0.1.4
pip install transformers==4.41.2
pip install huggingface-hub==0.23.4
pip install sentence-transformers==2.5.1
pip install chromadb
pip install wget==3.2
pip install torch==2.3.1
```

## Configuration

### 1. API Setup
Before running the notebook, configure your IBM Watsonx credentials:

```python
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "api_key": "YOUR_ACTUAL_API_KEY_HERE"  # Replace with your API key
}

project_id = "YOUR_PROJECT_ID_HERE"  # Replace with your project ID
```

**Important**: Never commit your API keys to version control. Use environment variables or secure key management.

### 2. Model Selection
Choose from available IBM Watsonx models:
- `ibm/granite-3-3-8b-instruct` (default)
- `mistralai/mistral-small-3-1-24b-instruct-2503`
- `meta-llama/llama-2-70b-chat`
- Or other supported models

## Usage

### Step 1: Prepare Your Document
Place your private document in the working directory or provide a URL for downloading.

### Step 2: Run the Notebook Cells
Execute the cells in order:

1. **Import Libraries**: Load all necessary dependencies
2. **Download/Load Document**: Specify your document path or URL
3. **Configure Model**: Set your API credentials and model parameters
4. **Process Document**: The system will:
   - Load and split the document
   - Create embeddings
   - Store in vector database
5. **Generate Summary**: Run the summarization query

### Example Usage
```python
# For local file
filename = 'your-private-document.txt'

# For remote file
url = 'https://your-secure-server.com/document.txt'

# Run summarization
query = "Summarize this document"
result = qa.invoke(query)
print(result['result'])
```

## Customization Options

### 1. Summarization Parameters
Adjust the generation parameters for different needs:
```python
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  # or SAMPLING
    GenParams.MIN_NEW_TOKENS: 100,    # Minimum summary length
    GenParams.MAX_NEW_TOKENS: 500,    # Maximum summary length
    GenParams.TEMPERATURE: 0.7,       # Creativity (0.1-1.0)
    GenParams.TOP_P: 0.9,             # Diversity control
    GenParams.TOP_K: 50               # Token selection
}
```

### 2. Document Processing
```python
# Adjust chunking for different document types
text_splitter = CharacterTextSplitter(
    chunk_size=1500,      # Larger for technical docs
    chunk_overlap=200,    # Maintain context
    separator="\n"        # Split by paragraphs
)
```

### 3. Custom Prompts
Modify the prompt template for specific requirements:
```python
prompt_template = """
Please provide a detailed summary focusing on:
1. Key findings
2. Main recommendations
3. Critical data points

Context: {context}
Question: {question}

Summary:
"""
```

## Advanced Features

### 1. Multiple Document Support
Extend to process multiple documents:
```python
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('./documents/', glob="*.txt")
documents = loader.load()
```

### 2. Conversation Memory
Enable follow-up questions:
```python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=flan_ul2_llm,
    retriever=docsearch.as_retriever(),
    memory=memory
)
```

### 3. Different Embedding Models
```python
from sentence_transformers import SentenceTransformer

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Alternative models
)
```

## Security Considerations

### For Private/Sensitive Documents:
1. **Local Processing**: Document text is processed locally
2. **No Data Retention**: IBM Watsonx may have data handling policies
3. **API Key Protection**: Secure your credentials
4. **Network Security**: Use HTTPS for document transfers

### Recommended Practices:
- Process highly sensitive documents entirely offline
- Review IBM's data privacy policies
- Implement audit logging
- Use secure storage for processed documents

## Troubleshooting

### Common Issues:

1. **API Authentication Error**
   - Verify API key is valid
   - Check project ID matches
   - Ensure service region is correct

2. **Document Loading Issues**
   - Check file encoding (UTF-8 recommended)
   - Verify file permissions
   - Ensure network connectivity for remote files

3. **Memory/Performance Issues**
   - Reduce chunk_size for large documents
   - Use CPU-optimized embeddings
   - Clear ChromaDB cache if needed

### Error Messages:
- `401 Unauthorized`: Invalid API credentials
- `404 Not Found`: Model or project doesn't exist
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Contact IBM support

## Model Alternatives

If IBM Watsonx is unavailable, consider:

### 1. Local LLMs with Ollama:
```python
# Requires Ollama installed locally
from langchain.llms import Ollama
llm = Ollama(model="llama2")
```

### 2. OpenAI API:
```python
from langchain.llms import OpenAI
llm = OpenAI(api_key="your-openai-key", temperature=0.7)
```

### 3. HuggingFace Models:
```python
from langchain.llms import HuggingFacePipeline
# Load local HuggingFace model
```

## Output Examples

### Sample Summarization:
```
Input: 50-page research document
Output: "This document presents a comprehensive analysis of market trends in Q4 2024. Key findings include: 1) 15% growth in renewable energy investments, 2) Emerging markets showing 25% higher returns, 3) Regulatory changes impacting tech sector. Recommendations focus on diversifying portfolios and increasing ESG compliance."
```

### Response Format:
The tool returns a dictionary containing:
- `result`: The generated summary
- `source_documents`: References to source chunks
- Additional metadata

## License and Attribution

### Dependencies:
- IBM Watsonx AI SDK
- LangChain framework
- ChromaDB vector database
- Sentence Transformers

### Usage Rights:
- Check IBM Watsonx terms of service
- Review model licensing agreements
- Ensure compliance with document copyrights

## Support and Resources

### Documentation:
- [IBM Watsonx Documentation](https://dataplatform.cloud.ibm.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Guide](https://docs.trychroma.com/)

### Community:
- IBM Developer Community
- LangChain Discord
- Stack Overflow tags: `ibm-watsonx`, `langchain`

### Getting Help:
1. Check the troubleshooting section
2. Review error logs
3. Consult IBM support for API issues
4. GitHub issues for code problems

## Updates and Maintenance

### Regular Updates Needed:
1. API library versions
2. Model compatibility
3. Security patches
4. Dependency updates

### Version Compatibility:
Tested with:
- Python 3.8-3.11
- LangChain 0.1.x
- IBM Watsonx AI 0.2.x

---

**Note**: Always test with non-sensitive documents first. Ensure you have appropriate rights to process and summarize documents. For production use, implement additional error handling, logging, and security measures.
