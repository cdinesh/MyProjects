TRACK_WORDS = ['Netflix','netflix','-filter:retweets']
TABLE_NAME = "All_Tweets_Combined"
TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at DATETIME, track VARCHAR(255), text VARCHAR(255) UNIQUE, original_text VARCHAR(255),\
            polarity INT, compound_score DOUBLE ,negative_score DOUBLE,positive_score DOUBLE,neutral_score DOUBLE,subjectivity INT, hashtags VARCHAR(255), \
            user_created_at VARCHAR(255), user_id VARCHAR(255), user_name VARCHAR(255), user_screen_name VARCHAR(255), user_location VARCHAR(255), \
            user_description VARCHAR(255), user_followers_count INT, user_friends_count INT, longitude DOUBLE, latitude DOUBLE, \
            retweet_count INT, favorite_count INT"
UNIQUE_COLUMNS = "text"
HISTORY_TABLE_NAME = "All_Tweets_History"
HISTORY_TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at DATETIME, track VARCHAR(255), text VARCHAR(255) UNIQUE, original_text VARCHAR(255),\
            polarity INT, compound_score DOUBLE ,negative_score DOUBLE,positive_score DOUBLE,neutral_score DOUBLE,subjectivity INT, hashtags VARCHAR(255), \
            user_created_at VARCHAR(255), user_id VARCHAR(255), user_name VARCHAR(255), user_screen_name VARCHAR(255), user_location VARCHAR(255), \
            user_description VARCHAR(255), user_followers_count INT, user_friends_count INT, longitude DOUBLE, latitude DOUBLE, \
            retweet_count INT, favorite_count INT"
