from flask import Flask, request
import os
import mysql.connector

app = Flask(__name__)

@app.route('/')
def index():
    return '''
        <html>
            <body>
              <form action="/upload" method="post" enctype="multipart/form-data">
    <br/>
    <label>Before</label><input type="file" name="before"><br/>
    <label>After</label> <input type="file" name="after"><br/>
    <input type="submit">
</form>


            </body>
        </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    folder_path = './data/Medetec_foot_ulcer/test/images'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    before_file = request.files['before']
    before_file_path = os.path.join(folder_path, before_file.filename)
    before_file.save(before_file_path)
    
    after_file = request.files['after']
    after_file_path = os.path.join(folder_path, after_file.filename)
    after_file.save(after_file_path)

    
    # Store the file paths to MySQL database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="npwt"
    )
    mycursor = mydb.cursor()

    sql = "INSERT INTO image (before_wound, after_wound) VALUES (%s, %s)"
    val = (before_file_path, after_file_path)
    mycursor.execute(sql, val)

    mydb.commit()
    mydb.close()

    # Run another script to process the uploaded files
    os.system('python main-db.py')
    
    
    
    return 'Files uploaded successfully!'

    

if __name__ == '__main__':
    app.run(debug=True)
