# This file runs during container build time to get model weights built into the container
import helper as hp

if __name__ == "__main__":
    hp.download_model("cb")