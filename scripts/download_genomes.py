import ftplib
import os
import pandas as pd
import random

# Made with AI

def download_genomes_from_bvbrc(genome_list, ftp_server = "ftp.bvbrc.org", remote_path = "/genomes/", local_directory = "data", nr_of_genomes_to_download = 10):

    # Ensure the local directory exists
    os.makedirs(local_directory, exist_ok=True)
    
    nr_of_downloaded_genomes = 0
    
    for genome_id in genome_list:
        if nr_of_downloaded_genomes == nr_of_genomes_to_download:
            # Stop when nr of desired genomes is reached
            print(f"Reached nr of desired genomes: {nr_of_downloaded_genomes}/{nr_of_genomes_to_download}")
            return
        # Relative path from script location; assumes 'downloads' is at same level as script
        local_file = os.path.join(local_directory,f'{genome_id}.fna')
        if not os.path.isfile(local_file):
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
                        ftp.cwd(remote_path)  # Adjust if different
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
                            nr_of_downloaded_genomes += 1
                    except ftplib.error_perm as e:
                        print(f"Error retrieving file: {e}")
                        os.remove(local_file)
                        raise e
                    
                   
            except ftplib.all_errors as e:
                print(f"FTP connection error: {e}")
        
    return

def get_list_of_genomes_to_download(file_path = "downloads/genome_summary"):
    
    genome_summary_path = file_path

    genome_summary = pd.read_csv(genome_summary_path,sep='\t')

    filtered_genome_summary = genome_summary[genome_summary['genome_status'] != 'Deprecated']
    genome_list = filtered_genome_summary.genome_id.to_list()

    random.shuffle(genome_list)

    return genome_list


if __name__ == "__main__":
    genome_list = get_list_of_genomes_to_download(file_path="downloads/genome_summary")
    download_genomes_from_bvbrc(genome_list=genome_list, nr_of_genomes_to_download=2000)
