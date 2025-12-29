import sys
sys.path.insert(0, 'src')
from models import PersonnelMaster, PersonMonth, JMESStatus, Gender

# Test valid creation
p = PersonnelMaster(age_start=30, gender=Gender.FEMALE)
assert p.to_dict()['gender'] == 'Female'

pm = PersonMonth(person_id='test', month=1, age=30.0)
assert pm.to_dict()['jmes_current'] == 'MFD'

# Test validation
try:
    PersonnelMaster(age_start=15)  # Should fail
    assert False, 'Should have raised'
except AssertionError:
    pass

print('Step 3: Models verified')
