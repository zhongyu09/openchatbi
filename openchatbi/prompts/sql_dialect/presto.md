# Rules for Presto SQL
- Use 'LIKE' instead of 'ILIKE' in the Presto SQL.
- If there is a 'GROUP BY' clause in the user query, you can use serial number(1,2,3..) instead of the column names in the 'GROUP BY' clause.
- When filter Array type dimension, use ARRAYS_OVERLAP, e.g. ARRAYS_OVERLAP(states, ARRAY['CA'])
- If you have to write two SQL statements, ensure to separate them with a semicolon `;`.
- If there is no 'limit' or 'top' count mentioned in user question, default use "LIMIT 10000" in the SQL query. 
- If the SQL you provide includes fuzzy matching filters (e.g., 'name LIKE ...'), you should apply a GROUP BY clause for this dimension to handle cases where multiple rows have similar names.
- If a table name is referenced multiple times in the SQL query (e.g., table_name.column), assign an alias to the table to simplify the query, such as (a.column).
- If you need to write an SQL statement that involves division between two columns, use CAST to convert the numerator to a double type and ensure the denominator is not zero. For example: CASE WHEN SUM(column2)=0 THEN 0 ELSE CAST(SUM(column1) AS DOUBLE) / SUM(column2) END

## Datetime filter related rules
- Please use "INTERVAL '7' DAY" instead of "INTERVAL '1' WEEK" in the presto sql you given.
- Do not use "DATE_SUB" or "DATE_ADD", You can use only datetime calculation like "NOW() - INTERVAL '1' DAY".
- Use `NOW()` instead of `CURRENT_DATE` when you need to get the current date.
- If user ask date range measured in days or months, you need to make sure the end date is included, example: "from 2025-03-12 to 2025-03-16", the condition should be `WHERE event_date >= timestamp '2025-03-12 00:00:00' and event_date < timestamp '2025-03-17 00:00:00'`
- If user ask for time range measured in hours, the end time is not included, example: "from 2025-03-12 00:00:00 to 2025-03-16 00:00:00", the condition should be `WHERE event_date >= timestamp '2025-03-12 00:00:00' and event_date < timestamp '2025-03-16 00:00:00'`
- If user ask for daily breakdown, make sure the whole day are correctly filtered, e.g. "last 7 days per day" should be `WHERE event_date >= DATE_TRUNC('day', (NOW() - INTERVAL '7' DAY)) AND event_date < DATE_TRUNC('day', NOW())`

## Rules for Timezone
### 1. Default Timezone
All event_date in the table are stored in **UTC**. If the user specifies a timezone (e.g., CET, PST), convert between timezones accordingly.
### 2. Timezone Conversion Syntax 
- Use `AT TIME ZONE` to convert event_date to other timezone. Example, to convert to CET: `event_date_expr AT TIME ZONE 'CET'`
- User `with_timezone` function to define a constant timestamp with timezone. Example, 2025-05-06 00:00 at CET: `with_timezone(timestamp '2025-05-06 00:00:00', 'CET')`
### 3. WHERE Clause Conversion
- If the user query includes absolute(constant) filters with a specific timezone, convert the timestamp with user timezone to UTC. Keep relative time filters unchanged.
  - Example: `WHERE event_date >= timestamp '2025-01-01 00:00:00' and event_date < NOW() - INTERVAL '1' DAY` ->
    `WHERE event_date >= with_timezone(timestamp '2025-01-01 00:00:00', CET') AT TIME ZONE 'UTC' AND event_date < NOW() - INTERVAL '1' DAY`
- Instruction when user ask for daily breakdown with timezone
  - If ask for relative date, the filter condition should use the date as "trunc date at that timezone first, then convert to UTC"
    - Example: "last 7 days per day in NY time" should be `AND event_date >= DATE_TRUNC('day', (NOW() - INTERVAL '7' DAY) AT TIME ZONE 'America/New_York') AT TIME ZONE 'UTC' AND event_date < DATE_TRUNC('day', NOW() AT TIME ZONE 'America/New_York') AT TIME ZONE 'UTC'`
  - If ask for absolute(constant) date, the filter condition should convert the 00:00 timestamp with user timezone to UTC
    - Example: "during 2025-05-04 and 2025-05-14 per day in NY time" should be `AND event_date >= with_timezone(timestamp '2025-05-04 00:00:00', 'America/New_York') AT TIME ZONE 'UTC' AND event_date < with_timezone(timestamp '2025-05-15 00:00:00', 'America/New_York') AT TIME ZONE 'UTC'`
### 4. SELECT Clause Conversion
- If applying a function f to a datetime column (e.g.date_trunc, format_datetime), convert the event_date from UTC to the user’s timezone before applying f, then cast the result as TIMESTAMP.
- Example: `SELECT f(event_date) AS event_date`-> `SELECT CAST(f(event_date AT TIME ZONE 'CET') AS TIMESTAMP) AS event_date`
### 5. Full Example
- User Question: "Show me hourly pv using table fact_table from 2025-01-01 to yesterday in CET"
- Generated SQL:
```
SELECT 
  CAST(date_trunc('hour', event_date AT TIME ZONE 'CET') AS TIMESTAMP) AS event_date,
  SUM(pv) AS "PV"
FROM fact_table
WHERE 
  event_date >= with_timezone(timestamp '2025-01-01 00:00:00', 'CET') AT TIME ZONE 'UTC'
  AND event_date < (NOW() - INTERVAL '1' DAY)
```

## Rules for Array Dimension
- Filtering: Use ARRAYS_OVERLAP 
  - When filter value in Array type dimension , use ARRAYS_OVERLAP, e.g. ARRAYS_OVERLAP(states, ARRAY['CA'])
- Flattening Arrays: Use CROSS JOIN UNNEST 
  - When filter Array type dimension items from a row to multi rows, use `CROSS JOIN UNNEST(COALESCE(NULLIF(id_array, ARRAY[]), ARRAY[-1])) AS t(id)`; MAKE SURE the untested alias has format like `t(id)`
  - Additionally, when the subquery uses CROSS JOIN UNNEST, do not sum the metrics for the total array items without group by the unnested id.
- Avoid UNNEST if the user didn’t request it 
  - If the user refers to an array-type dimension without specifying any particular item, or does not ask to expand it into individual elements, you should not use CROSS JOIN UNNEST.
