import requests
import pandas as pd





def get_species_name_from_genome_id(genome_id):

    esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=assembly&term={genome_id}[Assembly%20Accession]&retmode=json"
        )
    uid = requests.get(esearch).json()["esearchresult"]["idlist"]
    if not uid:
        raise ValueError(f"Assembly accession {genome_id} not found on NCBI")
    uid = uid[0]


    idlist = data["esearchresult"]["idlist"]
    if not idlist:
        print(f"No assemblies found for TaxID {taxid}")
        return
    
    assembly_id = idlist[0]
    print(f'Found assembly ID {assembly_id} for taxonomy id: {taxid}')

    esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={assembly_id}&retmode=json"
    resp2 = requests.get(esummary_url)
    resp2.raise_for_status()
    data2 = resp2.json()

    data2["result"][assembly_id]