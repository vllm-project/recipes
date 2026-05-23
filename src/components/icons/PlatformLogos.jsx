// Brand marks served from public/platform-logos/ as static SVG files.
// Sources: HuggingFace from Iconify (logos:hugging-face-icon), Modal from
// modal.com's official favicon, AWS from devicon. SageMaker reuses the AWS
// mark since no SageMaker-specific brand glyph exists; the wordmark color is
// theme-swapped because the original navy ink vanishes on dark backgrounds.

export function HuggingFaceIcon({ className = "" }) {
  // eslint-disable-next-line @next/next/no-img-element
  return <img src="/platform-logos/huggingface.svg" alt="" aria-hidden="true" className={className} />;
}

function ModalIcon({ className = "" }) {
  // eslint-disable-next-line @next/next/no-img-element
  return <img src="/platform-logos/modal.svg" alt="Modal" className={className} />;
}

function SageMakerIcon({ className = "" }) {
  return (
    <span className={`inline-block ${className}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/platform-logos/sagemaker.svg" alt="AWS SageMaker" className="w-full h-full block dark:hidden" />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/platform-logos/sagemaker-dark.svg" alt="AWS SageMaker" className="w-full h-full hidden dark:block" />
    </span>
  );
}

export const PLATFORM_LOGOS = {
  modal: ModalIcon,
  sagemaker: SageMakerIcon,
};

export function PlatformLogo({ id, className }) {
  const Component = PLATFORM_LOGOS[id];
  if (!Component) return null;
  return <Component className={className} />;
}
