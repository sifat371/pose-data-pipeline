import pickle
import zipfile
import numpy as np

zip_path = '/home/motion/Downloads/h36m_sh_conf_cam_source_final.pkl.zip'

def deep_inspect(path):
    print(f"📂 Loading {path}...")
    with zipfile.ZipFile(path, 'r') as z:
        pkl_filename = z.namelist()[0]
        with z.open(pkl_filename) as f:
            data = pickle.load(f)

    # Let's look at the 'train' set
    train_data = data['train']
    
    # 1. Check Consistency (Do lengths match?)
    n_joints = len(train_data['joint_2d'])
    n_sources = len(train_data['source'])
    print(f"\n📊 Statistics:")
    print(f"   Total Training Frames: {n_joints}")
    print(f"   Total Source Entries:  {n_sources}")
    
    if n_joints == n_sources:
        print("   ✅ Data is aligned (Parallel Arrays)")
    else:
        print("   ⚠️ WARNING: Data length mismatch!")

    # 2. Check Shapes
    first_pose = train_data['joint_2d'][0]
    print(f"\nMEasurements:")
    print(f"   Joint Array Shape: {np.shape(first_pose)}") 
    # Expecting (16, 2) or (17, 2)
    
    # 3. decode the Source Strings (CRUCIAL)
    # These usually tell us: Subject, Action, Camera, Frame
    print(f"\n🏷️  Source String Examples (First 3):")
    for i in range(3):
        print(f"   [{i}]: {train_data['source'][i]}")
        
    # Check if we have 'S1', 'S5', etc in there to identify subjects
    print(f"\n   Checking Subject 1 (S1) existence...")
    s1_count = sum(1 for x in train_data['source'] if 'S1' in str(x))
    print(f"   Entries containing 'S1': {s1_count}")

deep_inspect(zip_path)