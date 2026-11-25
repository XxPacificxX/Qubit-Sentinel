// Demo app for the PQC scanner: deliberately uses quantum-vulnerable crypto.
const crypto = require("crypto");
const jwt = require("jsonwebtoken");

const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", {
  modulusLength: 2048,
});

const ecdh = crypto.createECDH("prime256v1");
ecdh.generateKeys();

function issueToken(user) {
  return jwt.sign({ sub: user }, privateKey, { algorithm: "RS256" });
}

module.exports = { issueToken, publicKey, ecdh };
