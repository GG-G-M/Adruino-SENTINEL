import os
import pickle
import shutil
from pathlib import Path

# ============================================
# CONFIGURATION - Match main program settings
# ============================================
REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"
GESTURES_FILE = "hand_gestures.pkl"

# ============================================
# DATA RESET UTILITY
# ============================================

def backup_data():
    """Create backup of current data before deletion"""
    backup_folder = "backup_" + str(int(os.path.getmtime(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else 0))
    
    try:
        os.makedirs(backup_folder, exist_ok=True)
        
        # Backup face images
        if os.path.exists(REGISTERED_FOLDER):
            backup_faces = os.path.join(backup_folder, "authorized_faces")
            shutil.copytree(REGISTERED_FOLDER, backup_faces, dirs_exist_ok=True)
            print(f"✅ Backed up face images to: {backup_faces}")
        
        # Backup encodings
        if os.path.exists(ENCODINGS_FILE):
            shutil.copy2(ENCODINGS_FILE, os.path.join(backup_folder, ENCODINGS_FILE))
            print(f"✅ Backed up face encodings to: {backup_folder}")
        
        # Backup gestures
        if os.path.exists(GESTURES_FILE):
            shutil.copy2(GESTURES_FILE, os.path.join(backup_folder, GESTURES_FILE))
            print(f"✅ Backed up gestures to: {backup_folder}")
        
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def delete_all_data():
    """Delete all registered data"""
    deleted_items = []
    
    # Delete face images folder
    if os.path.exists(REGISTERED_FOLDER):
        try:
            shutil.rmtree(REGISTERED_FOLDER)
            deleted_items.append(f"✅ Deleted folder: {REGISTERED_FOLDER}")
        except Exception as e:
            deleted_items.append(f"❌ Failed to delete {REGISTERED_FOLDER}: {e}")
    
    # Delete encodings file
    if os.path.exists(ENCODINGS_FILE):
        try:
            os.remove(ENCODINGS_FILE)
            deleted_items.append(f"✅ Deleted file: {ENCODINGS_FILE}")
        except Exception as e:
            deleted_items.append(f"❌ Failed to delete {ENCODINGS_FILE}: {e}")
    
    # Delete gestures file
    if os.path.exists(GESTURES_FILE):
        try:
            os.remove(GESTURES_FILE)
            deleted_items.append(f"✅ Deleted file: {GESTURES_FILE}")
        except Exception as e:
            deleted_items.append(f"❌ Failed to delete {GESTURES_FILE}: {e}")
    
    return deleted_items

def delete_specific_user(username):
    """Delete a specific user's data"""
    deleted = []
    
    # Delete face image
    face_image = os.path.join(REGISTERED_FOLDER, f"{username}.jpg")
    if os.path.exists(face_image):
        try:
            os.remove(face_image)
            deleted.append(f"✅ Deleted face image: {face_image}")
        except Exception as e:
            deleted.append(f"❌ Failed to delete face image: {e}")
    
    # Remove from encodings
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                encodings_data = pickle.load(f)
            
            if username in encodings_data:
                del encodings_data[username]
                with open(ENCODINGS_FILE, 'wb') as f:
                    pickle.dump(encodings_data, f)
                deleted.append(f"✅ Removed from face encodings")
            else:
                deleted.append(f"⚠️  User not found in face encodings")
        except Exception as e:
            deleted.append(f"❌ Failed to update encodings: {e}")
    
    # Remove from gestures
    if os.path.exists(GESTURES_FILE):
        try:
            with open(GESTURES_FILE, 'rb') as f:
                gestures_data = pickle.load(f)
            
            if username in gestures_data:
                del gestures_data[username]
                with open(GESTURES_FILE, 'wb') as f:
                    pickle.dump(gestures_data, f)
                deleted.append(f"✅ Removed from gestures")
            else:
                deleted.append(f"⚠️  User not found in gestures")
        except Exception as e:
            deleted.append(f"❌ Failed to update gestures: {e}")
    
    return deleted

def list_current_data():
    """List all current registered data"""
    print("\n" + "="*60)
    print("  CURRENT REGISTERED DATA")
    print("="*60)
    
    # List face encodings
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                encodings_data = pickle.load(f)
            print(f"\n📸 Face Encodings: {len(encodings_data)} users")
            for name in encodings_data.keys():
                print(f"  • {name}")
        except Exception as e:
            print(f"❌ Error reading encodings: {e}")
    else:
        print("\n📸 Face Encodings: None")
    
    # List gestures
    if os.path.exists(GESTURES_FILE):
        try:
            with open(GESTURES_FILE, 'rb') as f:
                gestures_data = pickle.load(f)
            print(f"\n✋ Hand Gestures: {len(gestures_data)} users")
            for name in gestures_data.keys():
                print(f"  • {name}")
        except Exception as e:
            print(f"❌ Error reading gestures: {e}")
    else:
        print("\n✋ Hand Gestures: None")
    
    # List face images
    if os.path.exists(REGISTERED_FOLDER):
        images = [f for f in os.listdir(REGISTERED_FOLDER) if f.endswith('.jpg')]
        print(f"\n🖼️  Face Images: {len(images)} files")
        for img in images:
            print(f"  • {img}")
    else:
        print("\n🖼️  Face Images: None")
    
    print("="*60)

def get_data_statistics():
    """Get statistics about stored data"""
    stats = {
        'faces': 0,
        'gestures': 0,
        'images': 0,
        'total_size': 0
    }
    
    # Count face encodings
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                encodings_data = pickle.load(f)
            stats['faces'] = len(encodings_data)
            stats['total_size'] += os.path.getsize(ENCODINGS_FILE)
        except:
            pass
    
    # Count gestures
    if os.path.exists(GESTURES_FILE):
        try:
            with open(GESTURES_FILE, 'rb') as f:
                gestures_data = pickle.load(f)
            stats['gestures'] = len(gestures_data)
            stats['total_size'] += os.path.getsize(GESTURES_FILE)
        except:
            pass
    
    # Count images
    if os.path.exists(REGISTERED_FOLDER):
        images = [f for f in os.listdir(REGISTERED_FOLDER) if f.endswith('.jpg')]
        stats['images'] = len(images)
        for img in images:
            stats['total_size'] += os.path.getsize(os.path.join(REGISTERED_FOLDER, img))
    
    return stats

def main():
    print("\n" + "="*60)
    print("  DATA RESET UTILITY")
    print("  Security System Data Management")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("1. View Current Data")
        print("2. Delete Specific User")
        print("3. Delete ALL Data (with backup)")
        print("4. Delete ALL Data (NO backup) ⚠️  DANGEROUS")
        print("5. View Data Statistics")
        print("6. Exit")
        print("-"*60)
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == "1":
            list_current_data()
        
        elif choice == "2":
            list_current_data()
            username = input("\nEnter username to delete: ").strip()
            if username:
                confirm = input(f"⚠️  Delete '{username}'? (yes/no): ").lower()
                if confirm == 'yes':
                    results = delete_specific_user(username)
                    print()
                    for result in results:
                        print(result)
                    print(f"\n✅ User '{username}' deletion complete")
                else:
                    print("❌ Deletion cancelled")
            else:
                print("❌ Invalid username")
        
        elif choice == "3":
            list_current_data()
            stats = get_data_statistics()
            print(f"\n⚠️  WARNING: This will delete:")
            print(f"  • {stats['faces']} face encodings")
            print(f"  • {stats['gestures']} gestures")
            print(f"  • {stats['images']} images")
            print(f"  • Total size: {stats['total_size'] / 1024:.2f} KB")
            
            confirm = input("\nType 'DELETE ALL' to confirm: ").strip()
            if confirm == "DELETE ALL":
                print("\n📦 Creating backup...")
                if backup_data():
                    print("\n🗑️  Deleting all data...")
                    results = delete_all_data()
                    print()
                    for result in results:
                        print(result)
                    print("\n✅ All data deleted (backup created)")
                else:
                    print("❌ Backup failed, deletion cancelled")
            else:
                print("❌ Deletion cancelled")
        
        elif choice == "4":
            list_current_data()
            stats = get_data_statistics()
            print(f"\n⚠️  DANGER: This will permanently delete:")
            print(f"  • {stats['faces']} face encodings")
            print(f"  • {stats['gestures']} gestures")
            print(f"  • {stats['images']} images")
            print(f"  • NO BACKUP WILL BE CREATED")
            
            confirm = input("\nType 'DELETE FOREVER' to confirm: ").strip()
            if confirm == "DELETE FOREVER":
                results = delete_all_data()
                print()
                for result in results:
                    print(result)
                print("\n✅ All data permanently deleted")
            else:
                print("❌ Deletion cancelled")
        
        elif choice == "5":
            stats = get_data_statistics()
            print("\n" + "="*60)
            print("  DATA STATISTICS")
            print("="*60)
            print(f"Face Encodings: {stats['faces']}")
            print(f"Hand Gestures:  {stats['gestures']}")
            print(f"Face Images:    {stats['images']}")
            print(f"Total Size:     {stats['total_size'] / 1024:.2f} KB")
            print("="*60)
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()