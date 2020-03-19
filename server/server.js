// local HTTPS server for accessing camera on iOS device
// https://medium.com/@nitinpatel_20236/how-to-create-an-https-server-on-localhost-using-express-366435d61f28

const fs = require('fs');
const key = fs.readFileSync(__dirname + '/key.pem');
const cert = fs.readFileSync(__dirname + '/cert.pem');

const express = require('express');
const https = require('https');
const app = express();
const server = https.createServer({key: key, cert: cert }, app);
app.use(express.static(process.cwd() + '/src'));
app.get('/', (req, res) => {
  res.sendFile(process.cwd() + '/src/index.html');
});

server.listen(3001, () => { console.log('listening on 3001') });
