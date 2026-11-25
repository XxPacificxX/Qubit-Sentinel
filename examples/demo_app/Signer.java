// Demo app for the PQC scanner: deliberately uses quantum-vulnerable crypto.
import java.security.*;
import javax.crypto.KeyAgreement;

public class Signer {
    public static byte[] sign(byte[] data) throws Exception {
        KeyPairGenerator gen = KeyPairGenerator.getInstance("EC");
        gen.initialize(256);
        KeyPair kp = gen.generateKeyPair();
        Signature sig = Signature.getInstance("SHA256withECDSA");
        sig.initSign(kp.getPrivate());
        sig.update(data);
        return sig.sign();
    }

    public static KeyAgreement agreement() throws Exception {
        return KeyAgreement.getInstance("DH");
    }
}
