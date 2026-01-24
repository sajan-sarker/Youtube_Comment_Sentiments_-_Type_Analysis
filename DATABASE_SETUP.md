# Database Setup for Youtube Comment Sentiment & Type Analysis

This project requires a PostgreSQL database to store inference results and analysis history. Follow the steps below to set up the database connection.

## 1. Install PostgreSQL

- Download and install PostgreSQL from [https://www.postgresql.org/download/](https://www.postgresql.org/download/).
- Make sure to remember your username, password, database name, and port during installation.

## 2. Create the Results Table

Connect to your PostgreSQL database and create the required `results` table using the following SQL:

```
CREATE TABLE public.results
(
    id serial NOT NULL,
    result_type character varying(8) NOT NULL,
    comment character varying,
    url character varying,
    sentiment character varying(10),
    type_ character varying(10),
    sentiment_confidence double precision,
    type_confidence double precision,
    total_comments integer,
    sentiment_distribution_plot character varying,
    sentiment_confidence_plot character varying,
    type_distribution_plot character varying,
    type_confidence_plot character varying,
    date character varying NOT NULL,
    PRIMARY KEY (id)
);
```

## 3. Configure Environment Variables

The application reads database connection settings from environment variables (typically set in a `.env` file):

```
PS_HOST=localhost
PS_DB=your_database_name
PS_USER=your_postgres_user
PS_PASSWORD=your_password
```

Make sure this file is present in the project root, or set the environment variables manually.

## 4. Verify Connection

On running the FastAPI server (`main.py`), you should see a message:  
`Database connection successful`

You are now ready to use the project with a PostgreSQL backend!
