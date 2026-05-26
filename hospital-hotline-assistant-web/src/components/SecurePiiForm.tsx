import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { EmergencyPiiRequest, EmergencyPiiResponse } from '../api/types';

interface SecurePiiFormProps {
  /**
   * Async submit. Throws on backend validation failure / network
   * issues. Returns the redacted receipt with case_id + next
   * instruction on success.
   */
  onSubmit: (payload: EmergencyPiiRequest) => Promise<EmergencyPiiResponse>;
  /**
   * Optional triage badge to display above the form, e.g. when the
   * AI just classified the call as Level 1 (Red). Pure presentation.
   */
  triageLevel?: number;
  triageColor?: string;
  /**
   * Last receipt the parent has on file. Pass it in so the form can
   * stay rendered after a successful submission with the case ID
   * visible — instead of unmounting and losing the confirmation.
   */
  receipt?: EmergencyPiiResponse | null;
}

// Mirrors the Pydantic constraints on EmergencyPiiRequest in app/schemas.py
// so we surface errors locally before the round trip. The backend remains
// the source of truth — we never block submission purely on client rules.
const PHONE_PATTERN = /^[\d\s\-+().]{3,64}$/;
const NAME_MAX = 200;
const ADDRESS_MAX = 500;
const NOTES_MAX = 500;

interface FormErrors {
  name?: string;
  phone?: string;
  address?: string;
  notes?: string;
}

export function SecurePiiForm({
  onSubmit,
  triageLevel,
  triageColor,
  receipt,
}: SecurePiiFormProps) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [address, setAddress] = useState('');
  const [notes, setNotes] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);

  const submitted = Boolean(receipt);

  const validate = (): FormErrors => {
    const next: FormErrors = {};
    const trimmedName = name.trim();
    const trimmedPhone = phone.trim();
    const trimmedAddress = address.trim();
    const trimmedNotes = notes.trim();

    if (!trimmedName) next.name = t('piiErrorNameRequired');
    else if (trimmedName.length > NAME_MAX) next.name = t('piiErrorNameTooLong');

    if (!trimmedPhone) next.phone = t('piiErrorPhoneRequired');
    else if (!PHONE_PATTERN.test(trimmedPhone)) next.phone = t('piiErrorPhoneFormat');

    if (!trimmedAddress) next.address = t('piiErrorAddressRequired');
    else if (trimmedAddress.length > ADDRESS_MAX) next.address = t('piiErrorAddressTooLong');

    if (trimmedNotes.length > NOTES_MAX) next.notes = t('piiErrorNotesTooLong');

    return next;
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting || submitted) return;

    const localErrors = validate();
    setErrors(localErrors);
    if (Object.keys(localErrors).length > 0) return;

    setSubmitting(true);
    setServerError(null);
    try {
      await onSubmit({
        name: name.trim(),
        phone: phone.trim(),
        address: address.trim(),
        notes: notes.trim() || undefined,
      });
    } catch (err) {
      setServerError(err instanceof Error ? err.message : t('piiServerError'));
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted && receipt) {
    return (
      <section className="secure-pii-form receipt" aria-live="polite">
        <header className="secure-pii-form-header">
          <span className="secure-pii-form-icon" aria-hidden="true">
            ✓
          </span>
          <div>
            <h2>{t('piiSuccessTitle')}</h2>
            <p className="muted">{t('piiSuccessSubtitle')}</p>
          </div>
        </header>
        <dl className="secure-pii-receipt-grid">
          <div>
            <dt>{t('piiCaseId')}</dt>
            <dd>
              <code>{receipt.case_id}</code>
            </dd>
          </div>
          <div>
            <dt>{t('piiAlertStatus')}</dt>
            <dd>{receipt.alert_sent ? t('piiAlertDispatched') : t('piiAlertQueued')}</dd>
          </div>
        </dl>
        <p className="secure-pii-next-instruction">{receipt.next_instruction}</p>
      </section>
    );
  }

  return (
    <section className="secure-pii-form" aria-live="polite">
      <header className="secure-pii-form-header">
        <span className="secure-pii-form-badge" aria-hidden="true">
          🔒
        </span>
        <div>
          <h2>{t('piiFormTitle')}</h2>
          <p className="muted">{t('piiFormSubtitle')}</p>
        </div>
      </header>

      {triageLevel !== undefined && (
        <div
          className={`secure-pii-form-triage triage-${(triageColor ?? '').toLowerCase()}`}
          role="status"
        >
          <strong>
            {t('piiTriageLevelLabel', {
              level: triageLevel,
              color: triageColor ?? '',
            })}
          </strong>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        <div className="secure-pii-field">
          <label htmlFor="pii-name">
            {t('piiFieldName')} <span aria-hidden="true">*</span>
          </label>
          <input
            id="pii-name"
            type="text"
            inputMode="text"
            autoComplete="name"
            maxLength={NAME_MAX}
            value={name}
            onChange={(event) => setName(event.target.value)}
            aria-invalid={Boolean(errors.name)}
            aria-describedby={errors.name ? 'pii-name-error' : undefined}
            disabled={submitting}
            required
          />
          {errors.name && (
            <p id="pii-name-error" className="secure-pii-field-error">
              {errors.name}
            </p>
          )}
        </div>

        <div className="secure-pii-field">
          <label htmlFor="pii-phone">
            {t('piiFieldPhone')} <span aria-hidden="true">*</span>
          </label>
          <input
            id="pii-phone"
            type="tel"
            inputMode="tel"
            autoComplete="tel"
            maxLength={64}
            value={phone}
            onChange={(event) => setPhone(event.target.value)}
            aria-invalid={Boolean(errors.phone)}
            aria-describedby={errors.phone ? 'pii-phone-error' : undefined}
            disabled={submitting}
            required
          />
          {errors.phone && (
            <p id="pii-phone-error" className="secure-pii-field-error">
              {errors.phone}
            </p>
          )}
        </div>

        <div className="secure-pii-field">
          <label htmlFor="pii-address">
            {t('piiFieldAddress')} <span aria-hidden="true">*</span>
          </label>
          <textarea
            id="pii-address"
            rows={2}
            autoComplete="street-address"
            maxLength={ADDRESS_MAX}
            value={address}
            onChange={(event) => setAddress(event.target.value)}
            aria-invalid={Boolean(errors.address)}
            aria-describedby={errors.address ? 'pii-address-error' : undefined}
            disabled={submitting}
            required
          />
          {errors.address && (
            <p id="pii-address-error" className="secure-pii-field-error">
              {errors.address}
            </p>
          )}
        </div>

        <div className="secure-pii-field">
          <label htmlFor="pii-notes">{t('piiFieldNotes')}</label>
          <textarea
            id="pii-notes"
            rows={2}
            maxLength={NOTES_MAX}
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            aria-invalid={Boolean(errors.notes)}
            aria-describedby={errors.notes ? 'pii-notes-error' : undefined}
            placeholder={t('piiFieldNotesPlaceholder')}
            disabled={submitting}
          />
          {errors.notes && (
            <p id="pii-notes-error" className="secure-pii-field-error">
              {errors.notes}
            </p>
          )}
        </div>

        {serverError && (
          <p className="secure-pii-form-error" role="alert">
            {serverError}
          </p>
        )}

        <div className="secure-pii-form-actions">
          <button type="submit" className="primary-btn" disabled={submitting}>
            {submitting ? t('piiSubmitting') : t('piiSubmit')}
          </button>
          <p className="secure-pii-disclaimer muted">{t('piiDisclaimer')}</p>
        </div>
      </form>
    </section>
  );
}
