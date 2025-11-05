## 1 Installation

### 1.1 Create conda environment

```
conda create -n samcl python=3.10
conda activate samcl
```

### 1.2 Requirement

We  provided requirement.txt file to ensure environment compatibility:

```
pip install -r requirements.txt
```

### 1.3 Tools

During the data preparation phase, we will utilize the following three tools:

```
· PLIP
· Foldseek
· PDB-BRE
```

Additionally, since our model utilizes **SaProt** for feature extraction, it also requires downloading.  Your can follow these steps to install them：

#### 1.3.1 How to install PLIP

install:

```
git clone https://github.com/pharmai/plip.git
cd plip
pip install .
```

Suppose you have a complex structure file named complex.pdb (e.g., protein + peptide), you can run the following directly:

```
plip -f complex.pdb -o plip_output
```

#### 1.3.2 How to install Foldseek

install:

```
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz
export PATH=$(pwd)/foldseek/bin:$PATH
```

test:

```
foldseek
```

The output should be:

```
Foldseek  Version x.x.x
```

#### 1.3.3 How to install PDB-BRE

install for linux:

```
tar -xvzf PDB-BRE_1.0.0.tar.gz
cd PDB-BRE_1.0.0
```

To test the software installation, enter the following command:

```
cd <INSTALL_DIR>
./PDB-BRE-InterPair -h
```

The command to obtain binding site information is as follows:

```
./PDB-BRE-InterPair -f ./doc/peptide-protein.txt -t 'peptide' -c 50 -n 4
./PDB-BRE-DonSeqLabel -i ./Result/InterPairDeRedun_peptide_5.0_Any.csv
./PDB-BRE-ProSeqLabel -i ./Result/InterPairDeRedun_peptide_5.0_Any.csv -n 4
```

#### 1.3.4 How to install SaProt

install:

```
git clone https://github.com/westlake-repl/SaProt.git
cd SaProt/weights/PLMs
git clone https://huggingface.co/westlake-repl/SaProt_650M_PDB
```

## 2 Usage

We provide an example.csv file for reference, which you can view by running test.py. If you wish to train using your own data, you can replace the example.csv file using the data processing method we provide (./data).