# DS-Experiments

A data science experiments repository using DVC (Data Version Control) for managing datasets and experiments with Google Drive as remote storage.

## 📁 Project Structure

```
DS-Experiments/
├── .dvc/                    # DVC configuration files
│   ├── config              # DVC configuration
│   └── config.local        # Local DVC configuration (OAuth credentials)
├── data/                   # Data directory (tracked by DVC)
│   └── WholeTruthFoodDataset-combined.csv
├── data.dvc               # DVC file tracking the data directory
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Horus0305/DS-Experiments.git
cd DS-Experiments
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure DVC with Google Drive (OAuth Setup)

#### Step 1: Get OAuth Credentials from Google Cloud Console

1. **Go to Google Cloud Console**: Visit [Google Cloud Console](https://console.cloud.google.com/)

2. **Create or Select a Project**:
   - Create a new project or select an existing one
   - Make sure billing is enabled for the project

3. **Enable Google Drive API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"

4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - If prompted, configure the OAuth consent screen first:
     - Choose "External" user type
     - Fill in the required fields (App name, User support email, Developer contact email)
     - Add your email to test users
   - For Application type, choose "Desktop application"
   - Give it a name (e.g., "DVC Desktop Client")
   - Click "Create"

5. **Download Credentials**:
   - Download the JSON file containing your credentials
   - Keep this file secure and never commit it to version control

#### Step 2: Configure DVC with OAuth Credentials

1. **Configure OAuth credentials using DVC commands**:
   
   Replace `YOUR_CLIENT_ID` and `YOUR_CLIENT_SECRET` with the values from your downloaded JSON file:
   
   ```bash
   # Set the client ID (--local flag ensures it goes to config.local)
   dvc remote modify --local gdrive_remote gdrive_client_id YOUR_CLIENT_ID
   
   # Set the client secret (--local flag ensures it goes to config.local)
   dvc remote modify --local gdrive_remote gdrive_client_secret YOUR_CLIENT_SECRET
   ```

   Where:
   - `YOUR_CLIENT_ID` = the value of `client_id` field from your JSON file
   - `YOUR_CLIENT_SECRET` = the value of `client_secret` field from your JSON file

   **Important**: The `--local` flag ensures that credentials are stored in `.dvc/config.local` (which is not tracked by Git) instead of `.dvc/config` (which would be committed to the repository).

#### Step 3: Authenticate with Google Drive

```bash
# This will open a browser window for authentication
dvc pull
```

During the first `dvc pull`, you'll be redirected to a browser to:
1. Sign in to your Google account
2. Grant permission for DVC to access your Google Drive
3. The authentication token will be stored locally

### 5. Download Data

Once authentication is complete, download the data:

```bash
dvc pull
```

This will download the dataset from Google Drive to your local `data/` directory.

## 📊 Dataset Information

The repository contains the **WholeTruthFoodDataset-combined.csv** dataset, which is managed by DVC and stored on Google Drive for efficient version control and sharing.

## 🔧 Common DVC Commands

### Download latest data
```bash
dvc pull
```

### Upload data changes
```bash
dvc add data/
git add data.dvc .gitignore
git commit -m "Update dataset"
dvc push
```

### Check data status
```bash
dvc status
```

### Show data information
```bash
dvc data ls
```

## 🔒 Security Notes

- **Never commit OAuth credentials to Git**: The `config.local` file is automatically ignored by DVC
- **Keep your credentials secure**: Store the downloaded JSON file in a secure location
- **Regenerate credentials if compromised**: If your credentials are exposed, regenerate them in Google Cloud Console

## 🔍 Troubleshooting

### Authentication Issues
- If you get authentication errors, try: `dvc cache dir` to check cache location
- Clear authentication: Delete DVC cache and re-authenticate
- Make sure Google Drive API is enabled in your Google Cloud project

### Permission Issues
- Ensure your Google account has access to the shared Drive folder
- Check that the folder ID in `.dvc/config` is correct
- Verify that your OAuth app has the necessary scopes

### Data Access Issues
```bash
# Check DVC configuration
dvc config -l

# Verify remote configuration
dvc remote list

# Check data status
dvc status
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is open source. Please check with the repository owner for specific licensing terms.

## 📧 Contact

For questions or issues, please contact [Horus0305](https://github.com/Horus0305) or open an issue in this repository.
