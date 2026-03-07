import sys
import os

# --- PATH LOGIC ---
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

# 1. Base directory for WRITABLE files (stays outside EXE)
BASE_DIR = os.path.dirname(sys.executable if hasattr(sys, "_MEIPASS") else __file__)

# 2. Path to models (Inside EXE if bundled)
# This points to the FOLDER containing 'buffalo_l'
MODEL_HOME = resource_path("models")
os.environ["INSIGHTFACE_HOME"] = MODEL_HOME

# Define data paths
DB_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
NEW_PHOTOS = os.path.join(BASE_DIR, "new_photos")
PERSONS_DIR = os.path.join(BASE_DIR, "persons")
NO_FACE_DIR = os.path.join(BASE_DIR, "no_face_photos")
ERROR_DIR = os.path.join(BASE_DIR, "error_photos")
CSV_REPORT = os.path.join(BASE_DIR, "recognition_report.csv")

import cv2
import pickle
import shutil
import numpy as np
from insightface.app import FaceAnalysis
import warnings
import pandas as pd 
warnings.filterwarnings("ignore")
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import cv2

# ------------------ CONFIG ------------------
# DB_FILE = "embeddings.pkl"
# NEW_PHOTOS = "new_photos"
# PERSONS_DIR = "persons"
# NO_FACE_DIR = "no_face_photos"
# ERROR_DIR = "error_photos"
# CSV_REPORT = "recognition_report.csv"



# Configuration
SIMILARITY_THRESHOLD = 0.40  # InsightFace uses cosine similarity (0.3-0.5 typical)
MAX_EMBEDDINGS_PER_PERSON = 10
MIN_FACE_SIZE = 50  # Minimum face size in pixels
QUALITY_THRESHOLD = 0.3  # Minimum face quality (0-1)
face_log_data=[]

# ------------------ INITIALIZE MODEL ------------------
print("🔄 Loading InsightFace model...")
#3. INITIALIZE MODEL (Update this section)
print("🔄 Loading InsightFace model...")
app = FaceAnalysis(
    name="buffalo_l",
    root=BASE_DIR,  # Point to parent directory containing "models" folder
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ Model loaded!\n")
# ------------------ DB FUNCTIONS ------------------
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

# ------------------ FACE EMBEDDING ------------------
def extract_embedding(image_path):
    """Extract face embedding using InsightFace with quality checks"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ⚠️ Could not read image")
            return None, None
        
        # Detect faces
        faces = app.get(img) # gets faces in the image 
        
        if len(faces) == 0:
            print(f"   ⚠️ No face detected")
            return None, None
        
        # Get the largest/most confident face
        #    x.bbox[0] = left edge , x.bbox[1] = top edge , x.bbox[2] = top edge , x.bbox[3] = bottom edge 
        # lamda is basically area width*height   max() returns the faces having greatest pixel area and ignore other (closest to camera)
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # Quality checks
        bbox = face.bbox.astype(int) # gives boundings coordinates in int
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Check face size
        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            print(f"   ⚠️ Face too small: {face_width}x{face_height}px")
            return None, None
        
        # Check detection confidence
        det_score = face.det_score
        if det_score < QUALITY_THRESHOLD:
            print(f"   ⚠️ Low quality face (score: {det_score:.3f})")
            return None, None
        
        # Get embedding (512-dim normalized vector)
        embedding = face.normed_embedding
        
        # Additional face attributes (optional)
        age = int(face.age) if hasattr(face, 'age') else None  # if to  tackle crashing of code 

        gender = 'M' if face.gender == 1 else 'F' if hasattr(face, 'gender') else None
        
        print(f"   ✓ Embedding extracted (quality: {det_score:.3f}, size: {face_width}x{face_height})")
        if age:
            print(f"    Age: ~{age}, Gender: {gender}")
        
        return embedding, {
            'bbox': bbox,
            'quality': det_score,
            'age': age,
            'gender': gender,
            'size': (face_width, face_height)
        }
        
    except Exception as e:
        print(f"   ❌ Error extracting embedding: {str(e)}")
        return None, None

# ------------------ SIMILARITY ------------------
def cosine_similarity(a, b):
    """Calculate cosine similarity (InsightFace embeddings are normalized)"""
    return np.dot(a, b)

def find_match(embedding, db, threshold=SIMILARITY_THRESHOLD):
    """Find best matching person using multiple embeddings per person"""
    best_person = None
    best_score = -1  # Cosine similarity ranges from -1 to 1

    print("   🔍 Checking similarities:")
    for person, person_data in db.items():
        embeddings_list = person_data['embeddings']
        
        # Compare with all stored embeddings for this person
        scores = [cosine_similarity(embedding, stored_emb) 
                  for stored_emb in embeddings_list]
        max_score = max(scores)
        avg_score = np.mean(scores)
        
        print(f"      {person}: max={max_score:.3f}, avg={avg_score:.3f}")
        
        # Use max score for matching
        if max_score > threshold and max_score > best_score:
            best_score = max_score
            best_person = person

    if best_person:
        print(f"   ✅ Best match: {best_person} (score: {best_score:.3f})")
    else:
        print(f"   ❌ No match found (threshold: {threshold})")
    
    return best_person, best_score

# ------------------ PROCESS SINGLE PHOTO ------------------
def process_photo(image_path):
    """Process a single photo"""
    db = load_db()
    
    print(f"\n📸 Processing: {os.path.basename(image_path)}")
    
    # Extract embedding
    embedding, face_info = extract_embedding(image_path)

    # No face detected or poor quality
    if embedding is None:
        print("   🚫 Moving to no_face_photos/")
        os.makedirs(NO_FACE_DIR, exist_ok=True)
        shutil.move(
            image_path,
            os.path.join(NO_FACE_DIR, os.path.basename(image_path))
        )
        return

    # Find match
    person, score = find_match(embedding, db)

    # New person
    if person is None:
        existing_nums = [
            int(k.split("_")[1]) for k in db.keys() 
            if k.startswith("person_") and k.split("_")[1].isdigit()
        ]
        next_num = max(existing_nums, default=0) + 1
        person = f"person_{next_num}"
        
        # Initialize person data structure
        db[person] = {
            'embeddings': [embedding],
            'metadata': {
                'photo_count': 1,
                'avg_age': face_info['age'] if face_info['age'] else None,
                'gender': face_info['gender']
            }
        }
        print(f"   🆕 New person: {person}")
   
    # Existing person - add embeddin
    else:
        db[person]['embeddings'].append(embedding)
        db[person]['metadata']['photo_count'] += 1
        
        # Update average age if available
        if face_info['age'] and db[person]['metadata']['avg_age']:
            current_avg = db[person]['metadata']['avg_age']
            count = db[person]['metadata']['photo_count']
            db[person]['metadata']['avg_age'] = (current_avg * (count - 1) + face_info['age']) / count
        
        # Keep only last N embeddings to prevent bloat
        if len(db[person]['embeddings']) > MAX_EMBEDDINGS_PER_PERSON:
            db[person]['embeddings'] = db[person]['embeddings'][-MAX_EMBEDDINGS_PER_PERSON:]
        
        print(f"   ✅ Added to: {person} (total embeddings: {len(db[person]['embeddings'])})")
    face_log_data.append({
        "image_name":os.path.basename(image_path),
        "Identity":person,
        "~Age":face_info['age'],
        "Gender":face_info['gender'],
        "Confidence_Score":face_info['quality'],
        "Process_time":pd.Timestamp.now()
    })
    # Move photo to person's folder
    dest_dir = os.path.join(PERSONS_DIR, person)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.move(
        image_path,
        os.path.join(dest_dir, os.path.basename(image_path))
    )

    save_db(db)
    print(f"   ✓ Saved to {person}/")
# -------------------EXPORT FUNCTION------------------
def export_to_csv():
    """ Converts the collected list into a table and save it    """
    if not face_log_data:
        print("No face data to export")
        return
    #DATAFRAME CREATION : convverts our list of dictionaries into a table
    df=pd.DataFrame(face_log_data)
    # [5] CSV EXPORT: Saves the table. index=False removes the 0, 1, 2... numbering column
    df.to_csv(CSV_REPORT,index=True)
    print("Detailed Report CSV created ")
    return
# ------------------ PROCESS FOLDER ------------------
def is_folder_empty(folder):
    """Check if folder has no image files"""
    if not os.path.exists(folder):
        return True
    
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    return len(files) == 0

def process_folder(folder=NEW_PHOTOS):
    """Process all images in folder"""
    os.makedirs(folder, exist_ok=True)
    
    if is_folder_empty(folder):
        print(f"❌ No images found in '{folder}'. Please add photos to process.")
        return
    
    processed = 0
    errors = 0
    no_face = 0
    
    while not is_folder_empty(folder):
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not files:
            break

        image_name = files[0]
        image_path = os.path.join(folder, image_name)

        try:
            # Track results
            files_before = set(os.listdir(NO_FACE_DIR)) if os.path.exists(NO_FACE_DIR) else set()
            
            process_photo(image_path)
            
            files_after = set(os.listdir(NO_FACE_DIR)) if os.path.exists(NO_FACE_DIR) else set()
            
            if len(files_after) > len(files_before):
                no_face += 1
            else:
                processed += 1
                
        except Exception as e:
            print(f"   ❌ Critical error: {str(e)}")
            os.makedirs(ERROR_DIR, exist_ok=True)
            shutil.move(image_path, os.path.join(ERROR_DIR, image_name))
            errors += 1

    print(f"\n" + "="*50)
    print(f"✅ Processing complete!")
    print(f"   Successfully processed: {processed}")
    print(f"   No face detected: {no_face}")
    print(f"   Errors: {errors}")
    print("="*50)

# ------------------ MERGE PERSONS ------------------
def merge_persons(person1, person2):
    """Manually merge two person folders if they're the same person"""
    db = load_db()
    
    if person1 not in db or person2 not in db:
        print(f"❌ One or both persons not found in database")
        return
    
    # Merge embeddings
    db[person1]['embeddings'].extend(db[person2]['embeddings'])
    db[person1]['embeddings'] = db[person1]['embeddings'][-MAX_EMBEDDINGS_PER_PERSON:]
    
    # Update metadata
    db[person1]['metadata']['photo_count'] += db[person2]['metadata']['photo_count']
    
    # Move photos
    src_dir = os.path.join(PERSONS_DIR, person2)
    dest_dir = os.path.join(PERSONS_DIR, person1)
    
    if os.path.exists(src_dir):
        for photo in os.listdir(src_dir):
            shutil.move(
                os.path.join(src_dir, photo),
                os.path.join(dest_dir, photo)
            )
        os.rmdir(src_dir)
    
    # Remove from database
    del db[person2]
    save_db(db)
    
    print(f"✅ Merged {person2} into {person1}")

# ------------------ VIEW DATABASE STATS ------------------
def view_stats():
    """Display detailed database statistics"""
    db = load_db()
    persons = list(db.keys())
    
    print("\n" + "="*60)
    print("📊 Detailed Database Statistics")
    print("="*60)
    print(f"Total persons: {len(persons)}\n")
    
    if persons:
        for person in sorted(persons):
            person_dir = os.path.join(PERSONS_DIR, person)
            photo_count = 0
            
            if os.path.exists(person_dir):
                photo_count = len([f for f in os.listdir(person_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
            
            embedding_count = len(db[person]['embeddings'])
            metadata = db[person]['metadata']
            
            print(f"👤 {person}:")
            print(f"   Photos: {photo_count}")
            print(f"   Embeddings: {embedding_count}")
            
            if metadata.get('avg_age'):
                print(f"   Avg Age: ~{int(metadata['avg_age'])}")
            if metadata.get('gender'):
                print(f"   Gender: {metadata['gender']}")
            print()
    
    print("="*60)

# ------------------- Rename Person-------------------
def rename_person(old_name, new_name):
    """Rename a person in database, folder, and CSV report"""
    # update pickle database
    db = load_db()
    if old_name in db:
        db[new_name] = db.pop(old_name)
        save_db(db)
    
    # rename folder
    old_path = os.path.join(PERSONS_DIR, old_name)
    new_path = os.path.join(PERSONS_DIR, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    
    # update csv report
    if os.path.exists(CSV_REPORT):
        try:
            df = pd.read_csv(CSV_REPORT)
            if 'Identity' in df.columns:
                df.loc[df['Identity'] == old_name, 'Identity'] = new_name
                df.to_csv(CSV_REPORT, index=False)
        except Exception as e:
            print(f"Warning: Could not update CSV report: {e}")
    
    print(f"✅ Renamed {old_name} to {new_name} everywhere.")

#-------------------Deleting Person-------------------
def delete_person(name):
    #remove from pickle
    db=load_db()
    if name in db:
        del db[name]
        save_db(db)
    #2 delete from folder and images 
    folder_path=os.path.join("persons",name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)# deletes folder and images 
    #3 delete from csv report 
    if os.path.exists(CSV_REPORT):
        df = pd.read_csv(CSV_REPORT)
        df = df[df['Identity'] != name] # Remove all rows for this person
        df.to_csv(CSV_REPORT, index=False)

#-------------------Delete Individual Image-------------------
def delete_image(person_name, image_name):
    """Delete a single image from a person's folder"""
    image_path = os.path.join(PERSONS_DIR, person_name, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
        # Update CSV report
        if os.path.exists(CSV_REPORT):
            df = pd.read_csv(CSV_REPORT)
            df = df[~((df['Identity'] == person_name) & (df['image_name'] == image_name))]
            df.to_csv(CSV_REPORT, index=False)
        return True
    return False

#-------------------Get All Persons-------------------
def get_all_persons():
    """Get list of all persons with their photo counts - checks both database and filesystem"""
    db = load_db()
    persons_dict = {}
    
    # First, get persons from database
    for person_name in db.keys():
        person_dir = os.path.join(PERSONS_DIR, person_name)
        photo_count = 0
        first_image = None
        
        if os.path.exists(person_dir):
            try:
                image_files = [f for f in os.listdir(person_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
                photo_count = len(image_files)
                if image_files:
                    first_image = os.path.join(person_dir, image_files[0])
            except Exception as e:
                print(f"Warning: Error reading {person_dir}: {e}")
        
        persons_dict[person_name] = {
            'name': person_name,
            'photo_count': photo_count,
            'thumbnail': first_image
        }
    
    # Also check filesystem for person folders that might not be in database
    if os.path.exists(PERSONS_DIR):
        try:
            for folder_name in os.listdir(PERSONS_DIR):
                folder_path = os.path.join(PERSONS_DIR, folder_name)
                if os.path.isdir(folder_path) and folder_name not in persons_dict:
                    # This person exists in filesystem but not in database
                    try:
                        image_files = [f for f in os.listdir(folder_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
                        photo_count = len(image_files)
                        first_image = None
                        if image_files:
                            first_image = os.path.join(folder_path, image_files[0])
                        
                        persons_dict[folder_name] = {
                            'name': folder_name,
                            'photo_count': photo_count,
                            'thumbnail': first_image
                        }
                    except Exception as e:
                        print(f"Warning: Error reading {folder_path}: {e}")
        except Exception as e:
            print(f"Warning: Error reading {PERSONS_DIR}: {e}")
    
    # Return sorted list
    return sorted(persons_dict.values(), key=lambda x: x['name'])

#-------------------Get Person Images-------------------
def get_person_images(person_name):
    """Get all images for a specific person"""
    person_dir = os.path.join(PERSONS_DIR, person_name)
    if not os.path.exists(person_dir):
        return []
    
    image_files = [f for f in os.listdir(person_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    return [os.path.join(person_dir, img) for img in sorted(image_files)]

# ------------------ MAIN EXECUTION ------------------
if __name__ == "__main__":
    print("🚀 Starting InsightFace Recognition System")
    print(f"   Model: buffalo_l (ArcFace)")
    print(f"   Threshold: {SIMILARITY_THRESHOLD}")
    print(f"   Min Face Size: {MIN_FACE_SIZE}px\n")
    
    process_folder()
    
   # export_to_csv()
    # Display final statistics
    #view_stats()
    
    print("\n💡 Tips:")
    print("   • Lower threshold (0.3) = stricter matching, more separate persons")
    print("   • Higher threshold (0.5) = looser matching, fewer separate persons")
    print("   • Use merge_persons('person_1', 'person_2') to combine duplicates")
    print("   • Use view_stats() anytime to see detailed statistics")
