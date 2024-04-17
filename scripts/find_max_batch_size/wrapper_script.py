import subprocess
import sys

def run_pytorch_script(script_path, batch_size):
    # Build the command to run the PyTorch script with the provided batch size
    command = ['python', script_path, str(batch_size)]
    
    try:
        # Run the command and capture standard output and standard error
        result = subprocess.run(command, text=True, capture_output=True)
        
        # Check if there is any output in stderr
        if result.stderr:
            # Print error message if an error occurred
            print("Error output from PyTorch script:")
            print(result.stdout)
            print('='*60)
            print(result.stderr)
            # Check for a specific out-of-memory error
            if "CUDA out of memory" in result.stderr:
                print("The script failed due to an out-of-memory error.")
            else:
                print("The script encountered an error, but it is not an out-of-memory issue.")
        else:
            # Print standard output if the script ran successfully
            print("Script output:")
            print(result.stdout)
            print("The script ran successfully without any errors.")
            
    except Exception as e:
        print(f"An exception occurred while running the script: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python wrapper_script.py <path_to_pytorch_script> <batch_size>")
        sys.exit(1)

    # Get the script path and batch size from command line arguments
    script_path = sys.argv[1]
    batch_size = sys.argv[2]

    # Call the function to run the PyTorch script
    run_pytorch_script(script_path, batch_size)
