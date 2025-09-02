import ftplib
import os

# Made with AI

# Define the FTP server details
ftp_server = "ftp.bvbrc.org"
remote_path = "/RELEASE_NOTES/genome_summary"
local_directory = "downloads"  # Relative path from script location; assumes 'downloads' is at same level as script
local_file = os.path.join(local_directory, "genome_summary")

# Ensure the local directory exists
os.makedirs(local_directory, exist_ok=True)

try:
    # Connect to the FTP server and login anonymously if allowed
    with ftplib.FTP(ftp_server) as ftp:
        print("Connected to FTP server")
        
        # Provide credentials (anonymous or real) for authentication
        try:
            ftp.login()  # Anonymous login, you can replace with your credentials if needed
        except ftplib.all_errors as e:
            print(f"Failed to log in: {e}")
            raise e
        
        # Change the working directory; adjust this based on actual FTP server structure
        try:
            ftp.cwd("/RELEASE_NOTES")  # Adjust if different
        except ftplib.error_perm as e:
            print(f"Unable to change directory: {e}")
            raise e
        
        # Retrieve the file from the remote path
        try:
            with open(local_file, 'wb') as f:
                ftp.retrbinary(f"RETR {remote_path}", f.write)
                print(f"Downloaded {remote_path} to {local_file}")
        except ftplib.error_perm as e:
            print(f"Error retrieving file: {e}")
            raise e
except ftplib.all_errors as e:
    print(f"FTP connection error: {e}")
