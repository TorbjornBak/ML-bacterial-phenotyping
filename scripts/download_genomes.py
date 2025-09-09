import ftplib
import os

# Made with AI

def download_genomes_from_bvbrc(genome_list, ftp_server = "ftp.bvbrc.org", remote_path = "/genomes/", local_directory = "data"):

    # Ensure the local directory exists
    os.makedirs(local_directory, exist_ok=True)

    for genome_id in genome_list:
        # Relative path from script location; assumes 'downloads' is at same level as script
        local_file = os.path.join(local_directory,f'{genome_id}.fna')
        #print(local_file)
        try:
            # Connect to the FTP server and login anonymously if allowed
            with ftplib.FTP(host = ftp_server) as ftp:
                print("Connected to FTP server")
                
                # Provide credentials (anonymous or real) for authentication
                try:
                    ftp.login()  # Anonymous login, you can replace with your credentials if needed
                except ftplib.all_errors as e:
                    print(f"Failed to log in: {e}")
                    raise e
                
                # Change the working directory; adjust this based on actual FTP server structure
                try:
                    ftp.cwd("/genomes/")  # Adjust if different
                except ftplib.error_perm as e:
                    print(f"Unable to change directory: {e}")
                    raise e
                
    
                # Retrieve the file from the remote path
                try:
                    with open(local_file, 'wb') as f:
                        genome_path = f'{genome_id}/{genome_id}.fna'
                        print(genome_path)
                        ftp.retrbinary(f"RETR {genome_path}", f.write)
                        print(f"Downloaded {genome_path} to {local_file}")
                except ftplib.error_perm as e:
                    print(f"Error retrieving file: {e}")
                    raise e
        except ftplib.all_errors as e:
            print(f"FTP connection error: {e}")


if __name__ == "__main__":
    download_genomes_from_bvbrc(genome_list=[469009.4,
 1309411.5,
 1123738.3,
 551115.6,
 1856298.3,
 1706000.3,
 28901.2925,
 28901.2926,
 28901.2927,
 28901.2928]
)