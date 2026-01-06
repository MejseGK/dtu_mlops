from dotenv import load_dotenv
load_dotenv()
import os
print(os.environ["MY_VAR"])
print(os.environ["MY_OTHER_VAR"])
