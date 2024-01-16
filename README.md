# **DocumentGPT**

**Demo URL:** [https://hts-final-project-documentgpt.streamlit.app](https://hts-final-project-documentgpt.streamlit.app)

## **Introduction**
DocumentGPT is an innovative chatbot application for documents, allowing users to upload various file types (PDF, DOCX, ZIP) and interact with them through a chat interface. This README provides instructions on setting up and running DocumentGPT locally.

## **Prerequisites**
Before running the application, ensure you have the following installed:
 - Python (3.10 or later)
 - Pip (Python package manager)

## **Installation**

### Clone the Repository
Clone the DocumentGPT repository to your local machine using Git:
```
git clone [repo_url]
```

### Navigate to the Project Directory
After cloning, navigate to the DocumentGPT directory:
```
cd <directory_name>
```

### Install Dependencies
Install the necessary Python packages using pip:
```
pip install -r requirements.txt
```

## **Setting Up Environment Variables**

### Create a ```.streamlit``` Directory
Within the root of the project, create a ```.streamlit``` directory:
```mkdir .streamlit```

### Create a ```secrets.toml``` File
Inside the ```.streamlit``` directory, create a ```secrets.toml``` file to store your environment variables:
```
cd .streamlit
touch secrets.toml
```

### Configure ```secrets.toml```
Open ```secrets.toml``` in a text editor and add your environment variables. For instance:
```
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
QDRANT_URL="YOUR_QDRANT_URL"
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_ENV="YOUR_PINECONE_ENV"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## **Running the Application**

### Start the Streamlit Server
Navigate back to the root directory of your project and run the Streamlit server:
```streamlit run app.py```

### Accessing the Application
The application will be hosted locally and can be accessed through a web browser. Streamlit typically runs on ```http://localhost:8501```. Check the command-line output for the exact URL.

## **Usage**
 - Once the application is running, use the interface to upload your documents (PDF, DOCX, ZIP).
 - After uploading, you can start interacting with your documents through the chat interface.

## **Contributing**
We welcome contributions to DocumentGPT! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## **Contact**
For any queries or support, please reach out to [shaheryary1996@gmail.com](mailto:shaheryary1996@gmail.com).