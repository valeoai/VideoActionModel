import subprocess
import sys
import os
from datetime import datetime

def run_pytorch_script(script_path):
    
    log_file_path = f"{os.getenv('HOME')}/finder_log.txt"

    start_time = datetime.now()
    
    command = ['python', *script_path]
    print(*command)
    
    try:
        # Run the command and capture standard output and standard error
        result = subprocess.run(command, text=True, capture_output=True)
        
        # End time after running the script
        end_time = datetime.now()
        
        # Determine the type of outcome and write to log
        if result.stderr:
            # Print error message if an error occurred
            print("Error output from PyTorch script:")
            print(result.stdout)
            print('='*60)
            print(result.stderr)
            # Check for a specific out-of-memory error
            if "CUDA out of memory" in result.stderr:
                print("The script failed due to an out-of-memory error.")
                error_type = "ERR_OOM"
            else:
                print("The script encountered an error, but it is not an out-of-memory issue.")
                error_type = "ERR_OTHER"
        else:
            # Print standard output if the script ran successfully
            print("Script output:")
            print(result.stdout)
            print("The script ran successfully without any errors.")
            error_type = "SUCCESS"
            
        # Write log entry
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Start Time: {start_time}, End Time: {end_time}, Status: {error_type}\n")
            
    except Exception as e:
        # Log the exception
        with open(log_file_path, 'a') as log_file:
            end_time = datetime.now()
            log_file.write(f"Start Time: {start_time}, End Time: {end_time}, Status: ERR_OTHER\n")
        print(f"An exception occurred while running the script: {e}")

if __name__ == "__main__":

    print("/"*60)
    
    assert len(sys.argv) > 1, sys.argv

    # Get the script path and batch size from command line arguments
    script_path = sys.argv[1:]

    # Call the function to run the PyTorch script
    run_pytorch_script(script_path)
