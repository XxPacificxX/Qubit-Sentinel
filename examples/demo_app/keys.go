// Demo app for the PQC scanner: one vulnerable key, one post-quantum key exchange.
package main

import (
	"crypto/mlkem"
	"crypto/rand"
	"crypto/rsa"
)

func legacyKey() (*rsa.PrivateKey, error) {
	return rsa.GenerateKey(rand.Reader, 3072)
}

func pqKey() (*mlkem.DecapsulationKey768, error) {
	return mlkem.GenerateKey768()
}
