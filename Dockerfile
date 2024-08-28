# Επιλέγουμε το βασικό image
FROM python:3.9-slim

# Ορίζουμε τον τρέχοντα φάκελο εργασίας
WORKDIR /app

# Αντιγράφουμε τα αρχεία της εφαρμογής στο container
COPY . /app

# Εγκαθιστούμε τις εξαρτήσεις
RUN pip install -r requirements.txt

# Εκθέτουμε την πόρτα 5000 (προαιρετικά)
EXPOSE 5000

# Ορίζουμε την εντολή που θα εκτελείται κατά την εκκίνηση του container
CMD ["python", "app.py"]
