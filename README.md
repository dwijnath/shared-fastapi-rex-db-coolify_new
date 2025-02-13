# FastAPI Server for Connecting LLMs, AI Tools, or Frontends to Any Database

This FastAPI server provides a simple way to connect LLMs, AI tools, or frontends to any database. It includes multiple endpoints tailored for various use cases. For detailed explanations of each endpoint, visit the 'Build' section at [rex.tigzig.com](https://rex.tigzig.com).

---

### Build Command

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---

### Run Command

To start the server, execute:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

### Environment Variables and Database Connections

1. **RT_ENDPOINT and Proxy Server Setup**  
   - The RT_ENDPOINT is required for file upload functionality in the REX app
   - It serves as a proxy server to securely handle API calls to OpenAI, Gemini, or other LLM providers without exposing API keys
   - To set up the proxy server:
     ```bash
     git clone [PLACEHOLDER_REPO_URL] .
     pip install -r requirements.txt
     ```
   - The proxy server is a FastAPI application that needs to be running for file upload features to work

2. **Custom GPT Connections**  
   - Use the `connect-db` endpoint for custom GPTs to interact with any database. This does **not require** any environment variables. Simply deploy the app and point to this endpoint.

3. **Connecting Frontends, User Interfaces, or LLM Agents to Databases**  
   - If your app requires **file uploads**, the `RT_ENDPOINT` must be provided and the proxy server must be running
   - For apps that only need database connections to execute SQL queries, no environment variables are necessary

4. **Other Environment Variables**  
   - Additional variables (e.g., AWS, Azure, Neon, or Filessio credentials) may be used to hardcode database connections. These are **optional** and not needed for basic REX app functionality.


Here's a simplified description of the endpoints and their functionality for your README file:

---

## Endpoints Overview

This section provides a quick overview of the available endpoints, what they do, and the parameters they require. These endpoints allow connecting, uploading, and querying databases using FastAPI.

---

#### 1. **SQL Query Execution Endpoint**
**`GET /sqlquery/`**  
- **Description**: Executes a SQL query on a specified database. Supports `SELECT` and `non-SELECT` queries. Results from `SELECT` queries are returned as a text file.  
- **Parameters**:  
  - `sqlquery` (string): The SQL query to execute.  
  - `cloud` (string): The database provider (`azure`, `aws`, `neon`, `filessio`).  
- **Authentication**: Uses credentials from environment variables for the specified database provider. Useful if want to provide options across a set of hardcoded databases. 
- **Response**: Returns a text file for SELECT queries, or success status for non-SELECT queries.

---

#### 2. **Connect to Custom Database Endpoint**
**`GET /connect-db/`**  
- **Description**: Connects to a custom MySQL or PostgreSQL database using provided credentials and optionally executes a SQL query.  
- **Parameters**:  
  - `host` (string): Database host address.  
  - `database` (string): Name of the database to connect to.  
  - `user` (string): Database user name.  
  - `password` (string): Database password.  
  - `port` (integer, default=3306): Port number for the database.  
  - `db_type` (string, default=`mysql`): Database type (`mysql` or `postgresql`).  
  - `sqlquery` (string, optional): SQL query to execute.  
- **Authentication**: Accepts credentials as parameters.  Allows to connect to any database with just the credentials.

---

#### 3. **Upload File to LLM and PostgreSQL**
**`POST /upload-file-llm-pg/`**  
- **Description**: Uploads a file to generate a PostgreSQL table using an LLM for schema inference.  
- **Parameters**:  
  - `file` (file): The file to be uploaded.  
- **Additional Notes**:  
  - Uses the LLM for schema inference and creates a table in the `public` schema.  
  - Requires `RT_ENDPOINT` to be configured and the proxy server to be running
  - For connecting to hardcoded database connections. 

---

#### 4. **Upload File to LLM and MySQL**
**`POST /upload-file-llm-mysql/`**  
- **Description**: Uploads a file to generate a MySQL table using an LLM for schema inference.  
- **Parameters**:  
  - `file` (file): The file to be uploaded.  
- **Additional Notes**:  
  - Uses the LLM for schema inference and creates a table.  
  - Requires `RT_ENDPOINT` to be configured and the proxy server to be running
  - For connecting to hardcoded database connections. 

---

#### 5. **Upload File to Custom PostgreSQL Database**
**`POST /upload-file-custom-db-pg/`**  
- **Description**: Uploads a file to create a PostgreSQL table in a custom database using an LLM for schema inference.  
- **Parameters**:  
  - `host`, `database`, `user`, `password`, `port` (default=5432): Database connection details.  
  - `schema` (string, default=`public`): PostgreSQL schema for the table.  
  - `file` (file): The file to upload.  
- **Authentication**: Accepts database credentials as parameters.
- **Response**: Returns upload status, table name, row count, and performance metrics.

---

#### 6. **Upload File to Custom MySQL Database**
**`POST /upload-file-custom-db-mysql/`**  
- **Description**: Uploads a file to create a MySQL table in a custom database using an LLM for schema inference.  
- **Parameters**:  
  - `host`, `database`, `user`, `password`, `port` (default=3306): Database connection details.  
  - `sslmode` (string, optional): SSL mode for secure connections.  
  - `file` (file): The file to upload.  
- **Authentication**: Accepts database credentials as parameters.
- **Response**: Returns upload status, table name, row count, and performance metrics.


## Connecting to ChatGPT / CustomGP configuration
Sharing below the instructions and json schema for  ChatGPT configuration. This will allow your custom GPT to connect to any MySQL or PostgreSQL database.

### Instructions
Use this tool to connect to database . The user will provide host, database, username, password, and port as separate details or a URI (extract if needed). Use default ports (5432 for PostgreSQL, 3306 for MySQL) if unspecified. Use the database connection details shared by the user for all tool calls.

If any required information is missing, tell user what's missing. If all information is present, go ahead and try to connect and check for available schemas

IMPORTANT
1. Convert user questions into SQL queries (Postgres or MySQL compliant as per database type specified by user) and pass them as parameters in API calls. 
2. Ensure NO SCHEMA USED for MySQL queries as MySQL database does not have schemas. FOR MYSQL QUERIES DO NOT USE 'PUBLIC' OR ANY SCHEMA NAME. Queries to be MySQL compliant.
3. PostgreSQL has schemas. So postgres queries to always include schemas (use 'public' if unspecified). 

For errors, always share the query and connection details for debugging. Allow up to 180 seconds for query responses due to possible server delays. Always execute the query and share actual results, not fabricated data.


### OpenAPI JSON Schema
Shared in file `gptJson`
Currently it is pointing to the REX server. Feel free to use it for testing. You would need to replace the url with your own once you deploy your own FastAPI server.

### Help Guides  

Detailed guide on how to implement CustomGPT for Analytics Assistant, including how to create the OpenAPI JSON Schema. It uses example of a a different endpoint but the same logic can be applied to connect a custom GPT to your own FastAPI server.

[Analytics Assistant CustomGPT Implementation Guide](https://medium.com/@amarharolikar/analytics-assistant-customgpt-implementation-guide-9382887e95b5)  

For a detailed description of all the FastAPI endpoints and how they are implemented in the REX app, visit the 'Build' section at [rex.tigzig.com](https://rex.tigzig.com).


