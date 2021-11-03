from flask import Flask,request,jsonify,render_template,redirect
from model import predict
app = Flask(__name__)



@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files.get('file')

        if file is None or file.filename=="":
            return jsonify({'error':'no file'})

        img_bytes=file.read()
        val = predict(img_bytes)
        if len(val)<1:
            
            val="Error no face detected. Please submit a new image"
            isValid = False
        else:
            val = val[0]*0.453592
            isValid = True
        return render_template('index.html',val=val,isValid = isValid)
    else:
        return render_template('index.html')


if __name__ == "__main__" :
    app.run()