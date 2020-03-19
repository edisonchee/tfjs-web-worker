# Web Workers with TensorFlow.js

Example repo for my [article](https://edisonchee.com/writing/web-workers-with-tensorflow.js/).

Or, [launch the demo](https://edisonchee.github.io/tfjs-web-worker/).

## Getting started

If you want to run it on localhost, you'll need to setup a https server. Otherwise, `getUserMedia` will fail.

### Generate certs
```sh
openssl req -x509 -newkey rsa:2048 -keyout server/keytmp.pem -out server/cert.pem -days 365
```
```sh
openssl rsa -in server/keytmp.pem -out server/key.pem
```

### Start Express server
```sh
npm start
```