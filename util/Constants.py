# How many days worth of data do we include in an input?
DAYS_IN_AN_INPUT = 15

# How many data samples are in an input? If our input data is based on
# 3-hour intervals, we have 8 data samples a day. Then this value will be
# DAYS_IN_AN_INPUT * 8.
SAMPLES_OF_DATA_TO_LOOK_AT = DAYS_IN_AN_INPUT * 8
# If our data is based on 1 day intervals:
# SAMPLES_OF_DATA_TO_LOOK_AT = DAYS_IN_AN_INPUT
# If our data is based on 1-hour intervals:
# SAMPLES_OF_DATA_TO_LOOK_AT = DAYS_IN_AN_INPUT * 24
# If our data is based on 2-hour intervals:
# SAMPLES_OF_DATA_TO_LOOK_AT = DAYS_IN_AN_INPUT * 12

# How many channels do we have in an input? If we have price, volume, and 7
# indicators, we have 9 channels.
INPUT_CHANNELS = 9

# How many values are there in the output?
OUTPUT_CHANNELS = 7
