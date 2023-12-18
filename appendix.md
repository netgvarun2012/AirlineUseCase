# airasia Move Data Scientist Technical Assessment Appendix
v1.0.0

This document lays down the additional information you need to complete the assessment such as data dictionary.

## Appendix 1: Data dictionary for `hotel_sample_data_raw_bookings.csv`

|           Column name          |   Mode   |    Type   |                                                     Description                                                     |   |
|:---------------------------:|:--------:|:---------:|:-------------------------------------------------------------------------------------------------------------------:|---|
| hashed_order_number         | NULLABLE | INTEGER   | Unique identifier for every hotel order                                                                             |   |
| item_status                 | NULLABLE | STRING    | Order status, has been filtered for only confirmed booking, thus this column is useless in your case                |   |
| order_created_timestamp_utc | NULLABLE | TIMESTAMP | The timestamp recorded when order has been created in UTC timezone. Example value is 2023-06-17 14:01:38.000000 UTC |   |
| hashed_customer_id          | NULLABLE | INTEGER   | Unique identifier for customer                                                                                      |   |
| check_in_date               | NULLABLE | DATE      | Check-in date in YYYY-mm-dd format                                                                                  |   |
| check_out_date              | NULLABLE | DATE      | Check-out date in YYYY-mm-dd format                                                                                 |   |
| number_of_rooms             | NULLABLE | INTEGER   | Number of rooms booked                                                                                              |   |
| number_of_adult             | NULLABLE | INTEGER   | Number of adult for the hotel booked                                                                                |   |
| number_of_child             | NULLABLE | INTEGER   | Number of children for the hotel booked                                                                             |   |
| number_of_room_nights       | NULLABLE | INTEGER   | Number of room in nights                                                                                            |   |
| totalamount_usd             | NULLABLE | FLOAT     | Total amount of hotel cost for the booking in USD                                                                   |   |


## Appendix 2: Data dictionary for `hotel_sample_data_raw_inventories.csv`
The table contains dummy data for hotel inventories. `hotel_reference_id ` will be link key for the data Appendix 1 and Appendix 2.

|      Column name     |   Mode   |   Type  |                                       Description                                      |   |
|:--------------------:|:--------:|:-------:|:--------------------------------------------------------------------------------------:|---|
| hotel_reference_id   | NULLABLE | INTEGER | Unique identifier for hotel inventory                                                  |   |
| hotel_country        | NULLABLE | STRING  | Country of the hotel, only include THAILAND and PHILIPPINES                            |   |
| hotel_city           | NULLABLE | STRING  | City location of the hotel                                                             |   |
| hotel_state_province | NULLABLE | STRING  | State or province (administrative region 1 level after country) location of the hotel  |   |
| hotel_category       | NULLABLE | STRING  | Category of the hotel, either Hotel, Apartment or Guesthouse                           |   |


## Appendix 3: Data dictionary for `ride_sample_data_raw_bookings.csv`
The table contains dummy data for ride bookings from 2023-08-09 to 2023-09-09
|      Column name     |   mode   |    type   |                                       description                                      |   |
|:--------------------:|:--------:|:---------:|:--------------------------------------------------------------------------------------:|---|
| datetime_utc_booking | NULLABLE | TIMESTAMP | Timestamp when the ride booking is created, for example 2023-09-09 11:52:48.655000 UTC |   |
| pickup_latitude      | NULLABLE | FLOAT     | Latitude of pickup location                                                            |   |
| pickup_longitude     | NULLABLE | FLOAT     | Longitude of pickup location                                                           |   |