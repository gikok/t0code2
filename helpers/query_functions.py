from google.cloud import bigquery
import os
import re
import pandas as pd
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/gikok/code/ingka-search-modelling-dev-3a2a4f44d9c5.json'

client = bigquery.Client()

def get_ranked_queries(market: str, n_days: int, ranked_cols=True) -> pd.DataFrame:

    # Retrieves all events with click actions (event_action = select_content),their queries, resulting products, and number of queries resulting in specific product
    # Retrieves all events with click actions (event_action = select_content),their queries, and resulting products for the first PLP
    query = f"""
    WITH session_target AS (
    SELECT
        hit_sequence_number,
        event_category,
        event_action,
        session_id,
        page_url_query_parameter,
        next_page_url,
        SPLIT(REGEXP_EXTRACT(page_url_query_parameter, r'\?q=(\w.+)+'), '&')[SAFE_OFFSET(0)] AS clean_query,
        ARRAY_REVERSE(SPLIT(next_page_url,'/'))[SAFE_OFFSET(0)] AS target_product,
        ARRAY_REVERSE(SPLIT(ARRAY_REVERSE(SPLIT(next_page_url,'/'))[SAFE_OFFSET(0)], '-'))[SAFE_OFFSET(0)] AS target_product_id,
    FROM
        ingka-web-analytics-prod.web_data_v2.hits_events_and_pages
    WHERE
    ( 
        date_hit BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY) AND CURRENT_DATE()
        AND website_market_short = 'gb'
        AND event_category = 'engagement'
        AND event_action = 'select_content'
        AND page_url_query_parameter IS NOT NULL) 

    ),
    clean_query_product AS (
    SELECT
        session_id,
        clean_query,
        REGEXP_REPLACE(target_product, r'-', ' ') AS target_product,
        target_product_id,
        COUNT(*) AS query_n_resulting_product
    FROM
        session_target
    WHERE
        clean_query IS NOT NULL
        AND REGEXP_CONTAINS(target_product, r'[0-9A-Za-z]*[\d]')--target_product_id NOT IN ('products', 'en', 'Site Exit', 'gallery', 'shoppingcart', 'latest', 'search', 'login') -- prevent the target_product to become one of these 'next_pages'
    GROUP BY
        clean_query,
        target_product_id,
        target_product,
        session_id
    ORDER BY
        clean_query
    )
    SELECT 
        clean_query, 
        ARRAY_AGG(target_product_id IGNORE NULLS order by query_n_resulting_product desc) as term_frequency_ranking
        FROM clean_query_product 
        GROUP BY clean_query
    """

    df = client.query(query).to_dataframe()

    # adds the top 10 ranked query results as individual columns and drops the list
    if ranked_cols:
        
        df['rank_1'] = df['term_frequency_ranking'].apply(lambda x: re.sub("[^0-9]", "",list(x)[0]).zfill(8))
        
        for j in range(2, 11):
            df[f'rank_{j}'] = df['term_frequency_ranking'].apply(lambda x: re.sub("[^0-9]", "",list(x)[j-1]).zfill(8) if len(list(x))>(j-1) else None)
        
        df.drop(labels='term_frequency_ranking', axis=1, inplace=True)

    # some final cleaning that might have been missed by BQ
    df['clean_query'] = df['clean_query'].apply(lambda x: x.replace('%20', ' '))
    df['clean_query'] = df['clean_query'].apply(lambda x: x.replace('+', ' '))
    return df

def get_product_descriptions(market: str, language: str) -> pd.DataFrame:

    query = f"""
    SELECT 
        id, name, summary, benefits, key_w 
        FROM(
    (
        SELECT 
            GLOBAL_ID AS id,
            REGEXP_REPLACE(REGEXP_SUBSTR(REGEXP_SUBSTR(IMAGE_URL, "[^/]+$"),r"(\w.+)+__"),'-',' ') AS name,
            SUMMARY_TEXT_ENGLISH AS summary,
            BENEFIT_LIST AS benefits,
            TYPE_NAME_ENGLISH AS key_w
        FROM    
            `ingka-rrm-visualinthub-prod.visual_search_artefacts.global_running_range` as global
        WHERE
            IMAGE_URL LIKE '%/{language}/%'
            AND REGEXP_CONTAINS(TYPE_NAME_ENGLISH, r'[a-zA-Z]')
        ) 
        LEFT JOIN
            (SELECT GLOBAL_ID, LOCAL_ID FROM `ingka-rrm-visualinthub-prod.visual_search_artefacts.markets_running_range` WHERE COUNTRY_CODE = "{market}") as local
        ON 
            id = local.GLOBAL_ID
    )
    WHERE 
        GLOBAL_ID is not null
    """
    df = client.query(query).to_dataframe()

    # get item_no from item_id
    df['item_no'] = df['id'].apply(lambda x: re.sub("[^0-9]", "",x.split(",")[1]).zfill(8))

    # re-arrange and return dataframe
    df = df[['item_no']+list(df.columns[1:-1])]
    
    return df