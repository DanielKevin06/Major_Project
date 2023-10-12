import mysql.connector

# Connect to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="npwt"
)

# Get a cursor
mycursor = mydb.cursor()

# Insert the data into the database
sql = "INSERT INTO wound_healing (before_wound, after_wound, healing_percentage) VALUES (%s, %s, %s)"
val = ("n_white_pix1", "n_white_pix2", "wound_healing")
mycursor.execute(sql, val)

# Commit the changes
mydb.commit()

# Close the database connection
mydb.close()
