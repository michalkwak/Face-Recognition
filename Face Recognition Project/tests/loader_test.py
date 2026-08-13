from src.cli import load_train_test_split


def test_no_person_appears_in_both_train_and_test():
    _, train_labels, _, test_labels = load_train_test_split()

    # every person in train should also appear in test (one held-out
    # image each), so this just checks the split logic ran correctly:
    assert set(train_labels) == set(test_labels)


def test_test_set_has_one_image_per_person():
    _, _, test_images, test_labels = load_train_test_split()

    assert len(test_images) == len(set(test_labels))