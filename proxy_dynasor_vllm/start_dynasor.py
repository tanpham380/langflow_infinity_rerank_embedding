import sys
import patch_dynasor  # Import module patch để ghi đè hàm trước
from dynasor.cli.openai_server import main

if __name__ == "__main__":
    sys.exit(main())
