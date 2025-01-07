import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from web3 import Web3
import json
import os
import dotenv
import requests
import tempfile
import mimetypes
from urllib.parse import urlparse

os.environ.clear()
dotenv.load_dotenv(override=True)

# Ethereum Setup
base_rpc_url = os.getenv("BASE_RPC_URL")
web3 = Web3(Web3.HTTPProvider(base_rpc_url))
private_key = os.getenv("PRIVATE_KEY")
wallet_address = os.getenv("WALLET_ADDRESS")

# Contract Setup
abi = json.load(open("./contract_abi.json"))
contract_address = os.getenv("NFT_CONTRACT_ADDRESS")
nft_contract = web3.eth.contract(address=contract_address, abi=abi)

def mint_nft(description: str, image_url: str) -> dict:
    """
    Mint an NFT with the given attributes and image URL.
    
    Args:
        description (str): Description of the NFT
        image_url (str): URL of the NFT image
        
    Returns:
        dict: Contains status and transaction hash
    """
    if not contract_address:
        raise ValueError("Contract not configured")

    bucket = os.getenv("AWS_BUCKET")

    # Upload image to S3
    url = upload_image_from_url_to_s3(bucket, image_url)
    if not url:
        raise Exception("Failed to upload image to S3")
    print(f"Image URL: {url}")

    # Get the next token ID
    next_token_id = nft_contract.functions.tokenCounter().call()
    
    collection_name = os.getenv("NFT_COLLECTION_NAME")
    # Create and upload metadata
    metadata = {
        "name": f"{collection_name}{next_token_id}",
        "description": description,
        "image": url
   }
    
    metadata_url = upload_metadata_to_s3(bucket, next_token_id, metadata)
    if not metadata_url:
        raise Exception("Failed to upload metadata to S3")

    # Build transaction for minting
    gas_price = web3.eth.gas_price
    tx = nft_contract.functions.createNFT(wallet_address).build_transaction({
        'from': wallet_address,
        'nonce': web3.eth.get_transaction_count(wallet_address),
        'gas': 2000000,
        'gasPrice': gas_price,
        'chainId': 8453  # Base Mainnet Chain ID
    })

    # Sign and send transaction
    signed_tx = web3.eth.account.sign_transaction(tx, private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    return ("Minted", tx_hash.hex(), next_token_id)

def upload_image_from_url_to_s3(bucket_name, image_url, object_name=None):
    """
    Downloads an image from a URL and uploads it to an AWS S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        image_url (str): URL of the image to download and upload.
        object_name (str, optional): The name of the object in the bucket. Defaults to the image file name from URL.

    Returns:
        str: URL of the uploaded image if successful.
        None: If upload fails.
    """
    try:
        # Download the image from URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Get content type from response headers or guess from URL
        content_type = response.headers.get('Content-Type')
        if not content_type or not content_type.startswith('image/'):
            content_type = mimetypes.guess_type(image_url)[0]
        if not content_type:
            content_type = 'image/png'  # Default to png

        # Generate object name from URL if not provided
        if object_name is None:
            object_name = os.path.basename(urlparse(image_url).path)
            if not object_name or not object_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                object_name = f"image_{hash(image_url)}.png"  # Default to .png extension

        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Create an S3 client
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Upload the temporary file to S3 with content type
        s3_client.upload_file(
            temp_file_path, 
            bucket_name, 
            object_name,
            ExtraArgs={'ContentType': content_type}
        )

        # Clean up the temporary file
        os.unlink(temp_file_path)

        # Return the public URL of the image
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{object_name}"
        return url

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def upload_image_to_s3(bucket_name, image_path, object_name=None):
    """
    Uploads an image to an AWS S3 bucket using credentials from .env file.
    Can handle both local file paths and URLs.

    Args:
        bucket_name (str): The name of the S3 bucket.
        image_path (str): Local path to the image file or URL of the image.
        object_name (str, optional): The name of the object in the bucket.

    Returns:
        str: URL of the uploaded image if successful.
        None: If upload fails.
    """
    # Check if the input is a URL
    if image_path.startswith(('http://', 'https://')):
        return upload_image_from_url_to_s3(bucket_name, image_path, object_name)

    # Rest of the existing upload_image_to_s3 function remains the same
    if object_name is None:
        object_name = image_path.split("/")[-1]
        if not object_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            object_name += '.png'  # Add .png extension if no image extension present

    # Detect content type
    content_type = mimetypes.guess_type(image_path)[0]
    if not content_type:
        content_type = 'image/png'  # Default to png

    # Retrieve AWS credentials from environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    # Create an S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    try:
        # Upload the file to S3 with content type
        s3_client.upload_file(
            image_path, 
            bucket_name, 
            object_name,
            ExtraArgs={'ContentType': content_type}
        )
        print(f"Image uploaded successfully: {object_name}")

        # Return the public URL of the image
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{object_name}"
        return url

    except FileNotFoundError:
        print("The image file was not found.")
    except NoCredentialsError:
        print("AWS credentials not available.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials provided.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def upload_metadata_to_s3(bucket_name: str, token_id: int, metadata: dict) -> str:
    """
    Uploads NFT metadata as a JSON file to S3 using the token number as filename.
    
    Args:
        bucket_name (str): The name of the S3 bucket
        token_id (int): The token ID/number to use for the filename
        metadata (dict): The metadata to upload as JSON
        
    Returns:
        str: URL of the uploaded metadata JSON if successful
        None: If upload fails
    """
    try:
        # Create filename from token ID
        object_name = f"{token_id}.json"
        
        # Create an S3 client
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Create a temporary file to store the JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(metadata, temp_file, indent=2)
            temp_file_path = temp_file.name

        # Upload the JSON file to S3
        s3_client.upload_file(
            temp_file_path,
            bucket_name,
            object_name,
            ExtraArgs={
                'ContentType': 'application/json',
                'CacheControl': 'no-cache'  # Ensure metadata can be updated
            }
        )

        # Clean up the temporary file
        os.unlink(temp_file_path)

        # Return the public URL of the metadata
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{object_name}"
        print(f"Metadata uploaded successfully: {object_name}")
        return url

    except Exception as e:
        print(f"An error occurred uploading metadata: {e}")
        return None

