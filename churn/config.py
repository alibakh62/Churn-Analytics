# data range
START_DATE = '2018-05-01'
END_DATE = '2019-03-01'

# path to save spark ml models
MODELSPATH = ""

# settings
IMPUTE = "fill"
DATABASE = "test"

# Point Of Sale (POS) data
POS_RAW_DATA = ""  # transaction header data, i.e. basket level transactions
POS_TRANS_DETAIL = ""  # transaction detail data, product level transactions

# column names mapping
CUST_ID = ""  # customer loyalty ID
TIME_ID = "chunkid"
TRANS_AMT_COL = ""  # transaction amount
TRANS_ORD_COL = ""  # transaction ID
HOUR_OF_DAY_COL = ""
DAY_OF_WEEK_COL = ""
PSACODE_COL = ""  # product category code level 1
CATNAME_COL = ""  # product category name level 2
SUBCATNAME_COL = ""   # product category name level 3
TRANS_DATE = ""  # transaction date
CATCODE_COL = ""  # product category code level 2
SUBCATCODE_COL = ""  # product category code level 3
VENDORNAME_COL = ""  # product manufacturer name
IS_MEMBER = ""  # is member
PSANAME_COL = "" # product category name level 1

# features flag
GET_BRND_SEGMENT = True
GET_FEATURES = True
GET_BRND_FEATURES = True

# general features columns/flags
BASKET_DESC = True
DAY_OF_WEEK = True
HOUR_OF_DAY = True
MOST_COMMON = True
RFM_FEATURE = True
GROUPBY_COLS = ["memberid", "chunkid"]
MOSTCOMMON_COLS = ['state']

# brand-related columns/flags
BRAND_NAME = ""  # brand name as shown in data
BRAND_LOGIC_EXISTS = True
BRAND_BASKET_DESC = True
BRAND_KIT = True
BRAND_CESSATION = True
BRAND_MOSTCOMMON = True
BRAND_TRANS_AMT_COL = ""
BRAND_TRANS_ORD_COL = ""
BRAND_CAT = ""
BRAND_STARTERKIT = ""
BRAND_REFILLKIT = ""
BRAND_CESSATION = ""
BRAND_MOSTCOMMON_COLS = []
PSA12 = ""
PSA32 = ""
